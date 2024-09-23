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
#include "catalog/namespace.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "miscadmin.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "tcop/tcopprot.h"
#include "utils/datum.h"
#include "utils/memutils.h"

#include "external_index_socket.h"
#include "halfvec.h"
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
  MemoryContext oldCtx;
  HnswBuildState *buildstate = (HnswBuildState *)state;
  HnswGraph *graph = buildstate->graph;
  Vector *vec;
  usearch_label_t label;

#if PG_VERSION_NUM < 130000
  ItemPointer tid = &hup->t_self;
#endif

  /* Skip nulls */
  if (isnull[0])
    return;

  /* Use memory context */
  oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

  /* Insert tuple */
  vec = (Vector *)PG_DETOAST_DATUM(values[0]);

  label = ItemPointer2Label(tid);
  external_index_send_tuple(buildstate->external_socket, &label, vec->x,
                            buildstate->scalar_bits, vec->dim);
  SpinLockAcquire(&graph->lock);
  pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE,
                               ++graph->indtuples);
  SpinLockRelease(&graph->lock);
  /* Reset memory context */
  MemoryContextSwitchTo(oldCtx);
  MemoryContextReset(buildstate->tmpCtx);
}

static void InitUsearchIndexFromSocket(HnswBuildState *buildstate,
                                       usearch_init_options_t *opts,
                                       uint64 *nelem, uint64 *index_file_size) {

  usearch_error_t error = NULL;
  if (error != NULL) {
    elog(ERROR, "could not initialize usearch index");
  }

  buildstate->external_socket = palloc0(sizeof(external_index_socket_t));
  create_external_index_session(
      hnsw_external_index_host, hnsw_external_index_port,
      hnsw_external_index_secure, opts, buildstate, 100000);

  buildstate->reltuples = table_index_build_scan(
      buildstate->heap, buildstate->index, buildstate->indexInfo, true, true,
      ExternalIndexBuildCallback, (void *)buildstate, NULL);

  external_index_receive_metadata(buildstate->external_socket, nelem,
                                  index_file_size);
}

static void ImportExternalIndexInternal(Relation heap, Relation index,
                                        IndexInfo *indexInfo,
                                        HnswBuildState *buildstate,
                                        ForkNumber forkNum) {
  /* Read and Parse Usearch Index */
  MemoryContext oldCtx;
  MemoryContext tmpCtx;
  Buffer buf;
  Page page;
  BlockNumber blockno;
  BlockNumber last_data_block;
  metadata_t metadata;
  usearch_error_t error = NULL;
  usearch_init_options_t opts = {0};
  ldb_unaligned_slot_union_t *slots = NULL;
  Vector *vec = NULL;
  ItemPointerData tid;
  ItemPointerData entry_tid;
  ItemPointerData *item_pointers = NULL;
  OffsetNumber offset = 0;
  OffsetNumber maxoffset = 0;
  HnswElementTuple etup = NULL;
  HnswNeighborTuple ntup = NULL;
  HnswMetaPage metap = NULL;
  TupleDesc tupleDesc;
  Form_pg_attribute attr;

  Oid columnType;
  Oid Vector_Oid;
  Oid HalfVector_Oid;
  uint16 neighbor_len = 0;
  int32 i = 0;
  int32 j = 0;
  uint32 node_id = 0;
  uint32 node_level = 0;
  uint32 entry_level = 0;
  uint32 vector_bytes = 0;
  uint32 read_position = 0;
  uint32 node_size = 0;
  uint32 etupSize = 0;
  uint32 ntupSize = 0;
  uint32 combinedSize = 0;
  uint32 slot_count = 0;
  uint64 node_label = 0;
  uint64 total_read = 0;
  uint64 nelem = 0;
  uint64 index_file_size = 0;
  uint64 entry_slot = 0;
  uint64 seqid = 0;
  char *node = 0;
  char *external_index_data = NULL;
  char hdr_buffer[USEARCH_HEADER_SIZE];

  // unlogged table
  if (!heap)
    return;

  external_index_data = palloc0(EXTERNAL_INDEX_FILE_BUFFER_SIZE);
  tupleDesc = RelationGetDescr(heap);
  attr = &tupleDesc->attrs[index->rd_index->indkey.values[0] - 1];
  columnType = attr->atttypid;
  Vector_Oid = TypenameGetTypid("vector");
  HalfVector_Oid = TypenameGetTypid("halfvec");

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

  if (buildstate->procinfo->fn_addr == vector_negative_inner_product ||
      buildstate->procinfo->fn_addr == halfvec_negative_inner_product) {
    opts.metric_kind = usearch_metric_cos_k;
  } else if (buildstate->procinfo->fn_addr == vector_l2_squared_distance ||
             buildstate->procinfo->fn_addr == halfvec_l2_squared_distance) {
    opts.metric_kind = usearch_metric_l2sq_k;
  } else {
    elog(ERROR, "unsupported distance metric for external indexing");
  }

  buildstate->usearch_index = usearch_init(&opts, NULL, &error);
  if (error != NULL) {
    elog(ERROR, "failed to initialize usearch index");
  }
  metadata = usearch_index_metadata(buildstate->usearch_index, &error);

  if (error != NULL) {
    elog(ERROR, "failed to get usearch index metadata");
  }

  usearch_free(buildstate->usearch_index, &error);
  buildstate->usearch_index = NULL;

  InitUsearchIndexFromSocket(buildstate, &opts, &nelem, &index_file_size);
  /* ============== Parse Index END ============= */
  tmpCtx = AllocSetContextCreateInternal(
      CurrentMemoryContext, "HNSW External Context", 0,
      ALLOCSET_DEFAULT_INITSIZE,
      nelem * sizeof(ItemPointerData) + ALLOCSET_DEFAULT_MAXSIZE * 2);
  oldCtx = MemoryContextSwitchTo(tmpCtx);
  /* Create Metadata Page */
  CreateMetaPage(buildstate);
  /* =========== Create Metadata Page END ============= */

  elog(INFO, "indexed %zu elements", nelem);

  item_pointers = palloc(nelem * sizeof(ItemPointerData));

  total_read += external_index_read_all(
      buildstate->external_socket, (char *)&hdr_buffer, USEARCH_HEADER_SIZE);
  entry_slot = usearch_header_get_entry_slot((char *)&hdr_buffer);

  /* Append first index page */
  buf = ReadBufferExtended(index, forkNum, 0, RBM_NORMAL,
                           GetAccessStrategy(BAS_BULKREAD));
  LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
  page = BufferGetPage(buf);
  HnswBuildAppendPage(index, &buf, &page, forkNum);
  blockno = BufferGetBlockNumber(buf);
  /* ============ Append first page END =================  */

  total_read +=
      external_index_read_all(buildstate->external_socket, external_index_data,
                              EXTERNAL_INDEX_FILE_BUFFER_SIZE);

  vec = InitVector(buildstate->dimensions);
  etupSize = HNSW_ELEMENT_TUPLE_SIZE(VARSIZE_ANY(vec));
  etup = palloc0(etupSize);

  for (node_id = 0; node_id < nelem; node_id++) {
    // this function will add the tuples to index pages

    if ((EXTERNAL_INDEX_FILE_BUFFER_SIZE - read_position) < BLCKSZ &&
        total_read < index_file_size) {
      // rotate buffer
      memcpy(external_index_data, external_index_data + read_position,
             EXTERNAL_INDEX_FILE_BUFFER_SIZE - read_position);
      total_read += external_index_read_all(
          buildstate->external_socket,
          external_index_data +
              (EXTERNAL_INDEX_FILE_BUFFER_SIZE - read_position),
          read_position);
      read_position = 0;
    }

    node = external_index_data + read_position;
    node_level = level_from_node(node);
    node_size = node_tuple_size(node, metadata.dimensions, &metadata);
    node_label = label_from_node(node);

    /* Create Element Tuple */
    vector_bytes = node_vector_size(node, metadata.dimensions, &metadata);
    // there should not be an issue with mixing HalfVector and Vector types
    // as long as the struct layout is the same
    memcpy(vec->x, node + (node_size - vector_bytes), vector_bytes);
    // =======================================

    if (ntupSize !=
        HNSW_NEIGHBOR_TUPLE_SIZE(node_level, metadata.connectivity)) {

      ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(node_level, metadata.connectivity);
      if (ntup)
        pfree(ntup);

      ntup = palloc0(ntupSize);
    }

    combinedSize = etupSize + ntupSize + sizeof(ItemIdData);

    if (etupSize > HNSW_TUPLE_ALLOC_SIZE) {
      elog(ERROR, "index tuple too large");
    }

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
    ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;
    ntup->count = (node_level + 2) * metadata.connectivity;
    ntup->unused = 0;

    neighbor_len = 0;

    for (i = node_level; i >= 0; i--) {
      slot_count = 0;
      slots = get_node_neighbors_mut(&metadata, node, i, &slot_count);

      if (slot_count > ntup->count) {
        elog(ERROR,
             "neighbor list can not be more than %u in level %u, "
             "received %u",
             ntup->count, node_level, slot_count);
      }

      for (j = 0; j < slot_count; j++) {
        memcpy(&ntup->indextids[neighbor_len++].ip_blkid, &slots[j].seqid,
               sizeof(uint32));
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

    offset = PageAddItem(page, (Item)etup, etupSize, InvalidOffsetNumber, false,
                         false);

    ItemPointerSet(&tid, blockno, offset);

    if (node_id == entry_slot) {
      entry_level = node_level;
      memcpy(&entry_tid, &tid, sizeof(ItemPointerData));
    }

    memcpy(&item_pointers[node_id], &tid, sizeof(ItemPointerData));

    if (PageGetFreeSpace(page) < ntupSize) {
      HnswBuildAppendPage(index, &buf, &page, forkNum);
      blockno = BufferGetBlockNumber(buf);
    }

    offset = PageAddItem(page, (Item)ntup, ntupSize, InvalidOffsetNumber, false,
                         false);
    /* ================ Insert Tuples Into Index Page END =================== */

    // rotate buffer
    read_position += node_size;
  }

  pfree(vec);
  pfree(etup);

  // if there were no elements indexed ntup will not be allocated
  if (ntup)
    pfree(ntup);

  UnlockReleaseBuffer(buf);

  last_data_block = blockno;
  /* Update Entry Point */
  buf = ReadBufferExtended(index, forkNum, 0, RBM_NORMAL,
                           GetAccessStrategy(BAS_BULKREAD));
  LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
  page = BufferGetPage(buf);

  metap = HnswPageGetMeta(page);
  metap->insertPage = last_data_block;
  metap->entryBlkno = BlockIdGetBlockNumber(&entry_tid.ip_blkid);
  metap->entryOffno = entry_tid.ip_posid;
  metap->entryLevel = entry_level;

  MarkBufferDirty(buf);
  UnlockReleaseBuffer(buf);
  /* ============= Update Entry Point END ============== */

  /* Rewrite Neighbors */
  for (blockno = 1; blockno <= last_data_block; blockno++) {
    buf = ReadBufferExtended(index, forkNum, blockno, RBM_NORMAL,
                             GetAccessStrategy(BAS_BULKREAD));
    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
    page = BufferGetPage(buf);
    maxoffset = PageGetMaxOffsetNumber(page);

    for (offset = FirstOffsetNumber; offset <= maxoffset;
         offset = OffsetNumberNext(offset)) {
      HnswNeighborTuple neighborpage =
          (HnswNeighborTuple)PageGetItem(page, PageGetItemId(page, offset));
      if (!HnswIsNeighborTuple(neighborpage))
        continue;
      for (i = 0; i < neighborpage->count; i++) {
        if (!BlockNumberIsValid(
                BlockIdGetBlockNumber(&neighborpage->indextids[i].ip_blkid))) {
          continue;
        }

        seqid = 0;
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
  MemoryContextSwitchTo(oldCtx);
  MemoryContextDelete(tmpCtx);
}

void ImportExternalIndex(Relation heap, Relation index, IndexInfo *indexInfo,
                         HnswBuildState *buildstate, ForkNumber forkNum) {
  usearch_error_t error = NULL;
  PG_TRY();
  { ImportExternalIndexInternal(heap, index, indexInfo, buildstate, forkNum); }
  PG_CATCH();
  {
    if (buildstate->usearch_index) {
      usearch_free(buildstate->usearch_index, &error);
      buildstate->usearch_index = NULL;
    }

    if (buildstate->external_socket && buildstate->external_socket->close) {
      buildstate->external_socket->close(buildstate->external_socket);
    }
    PG_RE_THROW();
  }
  PG_END_TRY();
}
