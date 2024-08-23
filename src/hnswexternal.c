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

#include "external_index_socket.h"
#include "hnsw.h"
#include "storage/bufmgr.h"
#include "usearch.h"
#include "usearch_storage.hpp"
#include "utils/rel.h"
#include "vector.h"

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
  MemoryContext oldCtx;

#if PG_VERSION_NUM < 130000
  ItemPointer tid = &hup->t_self;
#endif

  /* Skip nulls */
  if (isnull[0])
    return;

  /* Insert tuple */
  Vector *vec = (Vector *)PG_DETOAST_DATUM(values[0]);

  usearch_label_t label = ItemPointer2Label(tid);
  external_index_send_tuple(buildstate->external_socket, &label, vec->x, 32,
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
      hnsw_external_index_secure, opts, 10000);

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

  opts.dimensions = buildstate->dimensions;
  opts.connectivity = buildstate->m;
  opts.expansion_add = buildstate->efConstruction;
  opts.pq = false;
  opts.multi = false;
  opts.quantization = usearch_scalar_f32_k;

  if (buildstate->procinfo->fn_addr == vector_negative_inner_product) {
    opts.metric_kind = usearch_metric_cos_k;
  } else if (buildstate->procinfo->fn_addr == vector_l2_squared_distance) {
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
    // TODO:::::: FIX usearch_get
    /*
    uint32 vector_bytes = node_vector_size( node, metadata.dimensions, &metadata
    ); Vector* vec = InitVector( buildstate->dimensions );
    // memcpy( vec->x, node, vector_bytes );
    float* vector = palloc0( vector_bytes );
    uint32 found_count = usearch_get( usearch_index, node_label, 1, vector,
    usearch_scalar_f32_k, &error ); elog( INFO, "FOUND %u", found_count );

    if( error != NULL ) {
       elog( ERROR, "error appeared %s", error );
    }

    if( found_count == 0 ) {
       elog( ERROR, "could not find vector" );
    }
    for( int i = 0; i < buildstate->dimensions; i++ ) {
       vec->x[ i ] = vector[ i ];
    }
    */
    // TODO::: temp solution of reading from heap to get vector
    memcpy(&element_tid, &node_label, sizeof(ItemPointerData));
    Buffer vector_buf = ReadBufferExtended(
        heap, forkNum, BlockIdGetBlockNumber(&element_tid.ip_blkid), RBM_NORMAL,
        GetAccessStrategy(BAS_BULKREAD));
    Page vector_page = BufferGetPage(vector_buf);
    TupleDesc vector_tuple_desc = RelationGetDescr(heap);
    ItemId vector_id = PageGetItemId(vector_page, element_tid.ip_posid);
    bool isNull;

    if (!ItemIdIsValid(vector_id)) {
      buildstate->external_socket->close(buildstate->external_socket);
      usearch_free(usearch_index, &error);
      elog(ERROR, "invalid item id");
    }

    HeapTupleData vector_tuple;
    vector_tuple.t_data = (HeapTupleHeader)PageGetItem(vector_page, vector_id);
    vector_tuple.t_len = ItemIdGetLength(vector_id);
    vector_tuple.t_tableOid = RelationGetRelid(heap);
    ItemPointerSet(&(vector_tuple.t_self),
                   BlockIdGetBlockNumber(&element_tid.ip_blkid),
                   element_tid.ip_posid);

    Datum vec_datum =
        heap_getattr(&vector_tuple, index->rd_index->indkey.values[0],
                     vector_tuple_desc, &isNull);

    if (isNull) {
      // There was a strange bug, when the build callback
      // was receiving newer version of tuple e.g (142, 6), but that tuple
      // was not visible here when trying to read from page.
      // ater FULL VACCUM it started to work, but the issue is not resolved.

      buildstate->external_socket->close(buildstate->external_socket);
      usearch_free(usearch_index, &error);
      elog(ERROR, "indexed element (%u, %u) can not be null",
           BlockIdGetBlockNumber(&element_tid.ip_blkid), element_tid.ip_posid);
    }
    Vector *vec = (Vector *)PG_DETOAST_DATUM(vec_datum);
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
    ReleaseBuffer(vector_buf);
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
