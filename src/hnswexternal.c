#include "postgres.h"

#include <math.h>

#include <string.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "miscadmin.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "tcop/tcopprot.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "utils/syscache.h"

#include "external_index_socket.h"
#include "hnsw.h"
#include "storage/bufmgr.h"
#include "usearch.h"
#include "usearch_storage.hpp"
#include "utils/rel.h"
#include "vector.h"
#include "halfvec.h"

#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#else
#include "pgstat.h"
#endif

#if PG_VERSION_NUM >= 130000
#define CALLBACK_ITEM_POINTER ItemPointer tid
#else
#define CALLBACK_ITEM_POINTER HeapTuple hup
#endif

static usearch_label_t ItemPointer2Label(ItemPointer itemPtr) {
  usearch_label_t label = 0;
  memcpy(&label, itemPtr, sizeof(*itemPtr));
  return label;
}
/*
 * Callback for table_index_build_scan
 */
static void ExternalIndexBuildCallback(Relation index, CALLBACK_ITEM_POINTER,
                                       Datum *values, bool *isnull,
                                       bool tupleIsAlive, void *state) {
  HnswBuildState *buildstate = (HnswBuildState *)state;
  HnswGraph *graph = buildstate->graph;

#if PG_VERSION_NUM < 130000
  ItemPointer tid = &hup->t_self;
#endif

  /* Skip nulls */
  if (isnull[0])
    return;

  /* Insert tuple */
  Vector *vec = (Vector *)PG_DETOAST_DATUM(values[0]);

  usearch_label_t label = ItemPointer2Label(tid);
  external_index_send_tuple(buildstate->external_socket, &label, vec->x, buildstate->scalar_bits,
                            vec->dim);
  SpinLockAcquire(&graph->lock);
  pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE,
                               ++graph->indtuples);
  SpinLockRelease(&graph->lock);
}

static void InitUsearchIndexFromSocket(HnswBuildState *buildstate,
                                       usearch_init_options_t *opts,
                                       char **data, uint64 *nelem) {

  usearch_error_t error = NULL;
  if (error != NULL) {
    elog(ERROR, "could not initialize usearch index");
  }

  buildstate->external_socket = create_external_index_session(
      hnsw_external_index_host, hnsw_external_index_port,
      hnsw_external_index_secure, opts, 100000);

  buildstate->reltuples = table_index_build_scan(
      buildstate->heap, buildstate->index, buildstate->indexInfo, true, true,
      ExternalIndexBuildCallback, (void *)buildstate, NULL);

  external_index_receive_index_file(buildstate->external_socket, nelem, data);
}

void ImportExternalIndex(Relation heap, Relation index, IndexInfo *indexInfo,
                         HnswBuildState *buildstate, ForkNumber forkNum) {
  /* Read and Parse Usearch Index */
  char *data;
  metadata_t metadata;
  usearch_index_t usearch_index;
  uint64 nelem = 0;
  usearch_error_t error = NULL;
  usearch_init_options_t opts = {0};

  TupleDesc tupleDesc = RelationGetDescr(heap);
  Form_pg_attribute attr = &tupleDesc->attrs[index->rd_index->indkey.values[0] - 1];
  Oid columnType = attr->atttypid;
  Oid Vector_Oid = GetSysCacheOid2(TYPENAMENSP,
						  Anum_pg_type_oid,
                          CStringGetDatum("vector"),
                          ObjectIdGetDatum(heap->rd_rel->relnamespace));
  Oid HalfVector_Oid = GetSysCacheOid2(TYPENAMENSP,
						  Anum_pg_type_oid,
                          CStringGetDatum("halfvec"),
                          ObjectIdGetDatum(heap->rd_rel->relnamespace));

  opts.dimensions = buildstate->dimensions;
  opts.connectivity = buildstate->m;
  opts.expansion_add = buildstate->efConstruction;
  opts.pq = false;
  opts.multi = false;

  if (columnType == Vector_Oid) {
	opts.quantization = usearch_scalar_f32_k;
	buildstate->scalar_bits = 32;
  } else if (columnType == HalfVector_Oid) {
	opts.quantization = usearch_scalar_f16_k;
	buildstate->scalar_bits = 16;
  } else {
    elog(ERROR, "unsupported element type for external indexing");
  }

  if (buildstate->procinfo->fn_addr == vector_negative_inner_product || buildstate->procinfo->fn_addr == halfvec_negative_inner_product) {
    opts.metric_kind = usearch_metric_cos_k;
  } else if (buildstate->procinfo->fn_addr == vector_l2_squared_distance || buildstate->procinfo->fn_addr == halfvec_l2_squared_distance) {
    opts.metric_kind = usearch_metric_l2sq_k;
  } else {
    elog(ERROR, "unsupported distance metric for external indexing");
  }

  usearch_index = usearch_init(&opts, NULL, &error);
  if (error != NULL) {
    elog(ERROR, "failed to initialize usearch index");
  }
  metadata = usearch_index_metadata(usearch_index, &error);

  if (error != NULL) {
    elog(ERROR, "failed to get usearch index metadata");
  }

  InitUsearchIndexFromSocket(buildstate, &opts, &data, &nelem);
  /* ============== Parse Index END ============= */

  /* Create Metadata Page */
  CreateMetaPage(buildstate);
  /* =========== Create Metadata Page END ============= */

  elog(INFO, "indexed %zu elements", nelem);

  ItemPointerData *item_pointers = palloc(nelem * sizeof(ItemPointerData));

  uint64 entry_slot = usearch_header_get_entry_slot(data);
  ItemPointerData entry_tid;
  ItemPointerData element_tid;
  uint64 progress = USEARCH_HEADER_SIZE;
  uint32 node_id = 0;
  uint32 node_level = 0;
  uint32 entry_level = 0;
  OffsetNumber offsetno = 0;
  uint32 node_size = 0;
  uint64 node_label = 0;
  char *node = 0;

  /* Append first index page */
  Buffer buf = ReadBufferExtended(index, forkNum, 0, RBM_NORMAL,
                                  GetAccessStrategy(BAS_BULKREAD));
  LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
  Page page = BufferGetPage(buf);
  HnswBuildAppendPage(index, &buf, &page, forkNum);
  BlockNumber blockno = BufferGetBlockNumber(buf);
  /* ============ Append first page END =================  */

  for (node_id = 0; node_id < nelem; node_id++) {
    // this function will add the tuples to index pages

    node = data + progress;
    node_level = level_from_node(node);
    node_size = node_tuple_size(node, metadata.dimensions, &metadata);
    node_label = label_from_node(node);

    /* Create Element Tuple */
    uint32 vector_bytes =
        node_vector_size(node, metadata.dimensions, &metadata);
    // there should not be an issue with mixing HalfVector and Vector types
    // as long as the struct layout is the same
    Vector *vec = InitVector(buildstate->dimensions);
    memcpy(vec->x, node + (node_size - vector_bytes), vector_bytes);
    // =======================================

    uint32 etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(vec));
    uint32 ntupSize =
        HNSW_NEIGHBOR_TUPLE_SIZE(node_level, metadata.connectivity);
    uint32 combinedSize = etupSize + ntupSize + sizeof(ItemIdData);

    if (etupSize > HNSW_TUPLE_ALLOC_SIZE) {
      buildstate->external_socket->close(buildstate->external_socket);
      usearch_free(usearch_index, &error);
      elog(ERROR, "index tuple too large");
    }

    HnswElementTuple etup = palloc0(etupSize);
    etup->type = HNSW_ELEMENT_TUPLE_TYPE;
    etup->level = node_level;
    etup->deleted = 0;

    for (int i = 0; i < HNSW_HEAPTIDS; i++) {
      if (i == 0)
        memcpy(&etup->heaptids[i], &node_label, sizeof(ItemPointerData));
      else
        ItemPointerSetInvalid(&etup->heaptids[i]);
    }

    memcpy(&etup->data, vec, VARSIZE_ANY(vec));
    /* ========= Create Element Tuple END ============ */

    /* Create Neighbor Tuple */
    HnswNeighborTuple ntup = palloc0(ntupSize);
    ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;
    ntup->count = (node_level + 2) * metadata.connectivity;
    ntup->unused = 0;

    uint16 neighbor_len = 0;

    for (int32 i = node_level; i >= 0; i--) {
      uint32 slot_count = 0;
      ldb_unaligned_slot_union_t *slots =
          get_node_neighbors_mut(&metadata, node, i, &slot_count);

      for (uint32 j = 0; j < slot_count; j++) {
        memcpy(&ntup->indextids[neighbor_len++].ip_blkid, &slots[j].seqid,
               sizeof(uint32));
        if (neighbor_len > ntup->count) {
          buildstate->external_socket->close(buildstate->external_socket);
          usearch_free(usearch_index, &error);
          elog(ERROR, "neighbor list can not be more than %u in level %u",
               ntup->count, node_level);
        }
      }
    }

    while (neighbor_len < ntup->count) {
      ItemPointerSetInvalid(&ntup->indextids[neighbor_len]);
      neighbor_len++;
    }
    /* ========= Create Neighbor Tuple END ============ */

    /* Insert Tuples Into Index Page */

    /* Keep element and neighbors on the same page if possible */
    if (PageGetFreeSpace(page) < etupSize ||
        (combinedSize <= HNSW_MAX_SIZE &&
         PageGetFreeSpace(page) < combinedSize))
      HnswBuildAppendPage(index, &buf, &page, forkNum);
    blockno = BufferGetBlockNumber(buf);

    if (combinedSize <= HNSW_MAX_SIZE) {
      ItemPointerSet(
          &etup->neighbortid, blockno,
          OffsetNumberNext(OffsetNumberNext(PageGetMaxOffsetNumber(page))));
    } else {
      ItemPointerSet(&etup->neighbortid, blockno + 1, FirstOffsetNumber);
    }

    ItemPointerData tid = {0};

    offsetno = PageAddItem(page, (Item)etup, etupSize, InvalidOffsetNumber,
                           false, false);

    ItemPointerSet(&tid, blockno, offsetno);

    if (node_id == entry_slot) {
      entry_level = node_level;
      memcpy(&entry_tid, &tid, sizeof(ItemPointerData));
    }

    memcpy(&item_pointers[node_id], &tid, sizeof(ItemPointerData));

    if (PageGetFreeSpace(page) < ntupSize) {
      HnswBuildAppendPage(index, &buf, &page, forkNum);
      blockno = BufferGetBlockNumber(buf);
    }

    offsetno = PageAddItem(page, (Item)ntup, ntupSize, InvalidOffsetNumber,
                           false, false);
    /* ================ Insert Tuples Into Index Page END =================== */

    progress += node_size;
  }

  UnlockReleaseBuffer(buf);

  BlockNumber last_data_block = blockno;
  /* Update Entry Point */
  buf = ReadBufferExtended(index, forkNum, 0, RBM_NORMAL,
                           GetAccessStrategy(BAS_BULKREAD));
  LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
  page = BufferGetPage(buf);
  HnswMetaPage metap = HnswPageGetMeta(page);

  metap->insertPage = last_data_block;
  metap->entryBlkno = BlockIdGetBlockNumber(&entry_tid.ip_blkid);
  metap->entryOffno = entry_tid.ip_posid;
  metap->entryLevel = entry_level;

  MarkBufferDirty(buf);
  UnlockReleaseBuffer(buf);
  /* ============= Update Entry Point END ============== */

  /* Rewrite Neighbors */
  for (BlockNumber blockno = 1; blockno <= last_data_block; blockno++) {
    buf = ReadBufferExtended(index, forkNum, blockno, RBM_NORMAL,
                             GetAccessStrategy(BAS_BULKREAD));
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
    page = BufferGetPage(buf);
    OffsetNumber maxoffset = PageGetMaxOffsetNumber(page);

    for (OffsetNumber offset = FirstOffsetNumber; offset <= maxoffset;
         offset = OffsetNumberNext(offset)) {
      HnswNeighborTuple neighborpage =
          (HnswNeighborTuple)PageGetItem(page, PageGetItemId(page, offset));
      if (!HnswIsNeighborTuple(neighborpage))
        continue;
      for (uint32 i = 0; i < neighborpage->count; i++) {
        if (!BlockNumberIsValid(
                BlockIdGetBlockNumber(&neighborpage->indextids[i].ip_blkid))) {
          continue;
        }

        uint32 seqid = 0;
        memcpy(&seqid, &neighborpage->indextids[i].ip_blkid, sizeof(uint32));
        memcpy(&neighborpage->indextids[i], &item_pointers[seqid],
               sizeof(ItemPointerData));
      }
    }

    MarkBufferDirty(buf);
    UnlockReleaseBuffer(buf);
  }
  /* =========== Rewrite Neighbors END ============= */

  buildstate->external_socket->close(buildstate->external_socket);
  usearch_free(usearch_index, &error);
}
