#include "postgres.h"

#include <string.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hnsw.h"
#include "storage/bufmgr.h"
#include "usearch.h"
#include "usearch_storage.hpp"
#include "vector.h"

void ImportExternalIndex( Relation heap, Relation index, IndexInfo* indexInfo, HnswBuildState* buildstate, ForkNumber forkNum, char* index_file_path )
{

   usearch_error_t error = NULL;
   usearch_init_options_t opts;
   struct stat index_file_stat;
   int index_file_fd;
   MemSet( &opts, 0, sizeof( opts ) );
   usearch_index_t usearch_index = usearch_init( &opts, NULL, &error );
   if( error != NULL ) {
      elog( ERROR, "could not load usearch index" );
   }

   usearch_load( usearch_index, index_file_path, &error );

   if( error != NULL ) {
      elog( ERROR, "failed to load index" );
   }

   metadata_t metadata = usearch_index_metadata( usearch_index, &error );
   if( error != NULL ) {
      elog( ERROR, "failed to read index metadata" );
   }

   index_file_fd = open( index_file_path, O_RDONLY );

   if( index_file_fd <= 0 ) {
      elog( ERROR, "failed to read index file" );
   }

   fstat( index_file_fd, &index_file_stat );
   char* data = mmap( NULL, index_file_stat.st_size, PROT_READ, MAP_PRIVATE, index_file_fd, 0 );

   if( data == MAP_FAILED ) {
      elog( ERROR, "failed to mmap index file" );
   }

   buildstate->m = metadata.connectivity;
   buildstate->efConstruction = metadata.expansion_add;
   CreateMetaPage( buildstate );

   uint32 nelem = usearch_size( usearch_index, &error );
   if( error != NULL ) {
      elog( ERROR, "failed to get index size" );
   }

   elog( INFO, "element count is %u", nelem );
   ItemPointerData* item_pointers = palloc( nelem * sizeof( ItemPointerData ) );

   uint32 node_id = 0;
   uint32 node_level = 0;
   uint32 entry_level = 0;
   uint64 entry_slot = 0;
   ItemPointerData entry_tid;
   elog( INFO, "Entry slot is %zu\n", entry_slot );
   uint32 node_size = 0;
   uint64 node_label = 0;
   uint64 progress = USEARCH_HEADER_SIZE;
   char* node = 0;

   Buffer buf = ReadBufferExtended( index, forkNum, 0, RBM_NORMAL, GetAccessStrategy( BAS_BULKREAD ) );
   LockBuffer( buf, BUFFER_LOCK_EXCLUSIVE );
   Page page = BufferGetPage( buf );
   OffsetNumber offsetno = 0;

   HnswBuildAppendPage(index, &buf, &page, forkNum);

   BlockNumber blockno = BufferGetBlockNumber( buf );

   entry_slot = usearch_header_get_entry_slot( data );

   for( node_id = 0; node_id < nelem; node_id++ ) {
      // this function will add the tuples to index pages

      node = data + progress;
      node_level = level_from_node( node );
      node_size = node_tuple_size( node, metadata.dimensions, &metadata );
      node_label = label_from_node( node );
      uint32 vector_bytes = node_vector_size( node, metadata.dimensions, &metadata );

      elog( INFO, "node size is %u, vector_bytes: %u, level is %u", node_size, vector_bytes, node_level );

      /* Create Element Tuple */
      // TODO:::::: FIX usearch_get
      /*
      Vector* vec = InitVector( buildstate->dimensions );
      // memcpy( vec->x, node, vector_bytes );
      float* vector = palloc0( vector_bytes );
      uint32 found_count = usearch_get( usearch_index, node_label, 1, vector, usearch_scalar_f32_k, &error );
      elog( INFO, "FOUND %u", found_count );

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
      // TODO::: temp solution of reading from heap
      ItemPointerData element_tid;
      memcpy( &element_tid, &node_label, sizeof( ItemPointerData ) );
      Buffer vector_buf = ReadBufferExtended( heap, forkNum, BlockIdGetBlockNumber( &element_tid.ip_blkid ), RBM_NORMAL, GetAccessStrategy( BAS_BULKREAD ) );
      Page vector_page = BufferGetPage( vector_buf );
      TupleDesc vector_tuple_desc = RelationGetDescr( heap );
      ItemId vector_id = PageGetItemId( vector_page, element_tid.ip_posid );
      bool isNull;

      if( !ItemIdIsValid( vector_id ) ) {
         elog( ERROR, "invalid item id" );
      }

      HeapTupleData vector_tuple;
      vector_tuple.t_data = (HeapTupleHeader)PageGetItem( vector_page, vector_id );
      vector_tuple.t_len = ItemIdGetLength( vector_id );
      vector_tuple.t_tableOid = RelationGetRelid( heap );
      ItemPointerSet( &( vector_tuple.t_self ), BlockIdGetBlockNumber( &element_tid.ip_blkid ), element_tid.ip_posid );

      Datum vec_datum = heap_getattr( &vector_tuple, 2, vector_tuple_desc, &isNull );
      Vector* vec = (Vector*)PG_DETOAST_DATUM( vec_datum );
      // =======================================

      uint32 etupSize = HNSW_ELEMENT_TUPLE_SIZE( VARSIZE_ANY( vec ) );
      uint32 ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE( node_level, metadata.connectivity );
      uint32 combinedSize = etupSize + ntupSize + sizeof( ItemIdData );

      HnswElementTuple etup = palloc0( etupSize );
      etup->type = HNSW_ELEMENT_TUPLE_TYPE;
      etup->level = node_level;
      etup->deleted = 0;
      for( int i = 0; i < HNSW_HEAPTIDS; i++ ) {
         if( i == 0 )
            memcpy( &etup->heaptids[ i ], &node_label, sizeof( ItemPointerData ) );
         else
            ItemPointerSetInvalid( &etup->heaptids[ i ] );
      }

      memcpy( &etup->data, vec, VARSIZE_ANY( vec ) );
      ReleaseBuffer( vector_buf );
      elog( INFO, "etupSize is %u, etup is %p, label is %zu, vector size is: %u", etupSize, etup, node_label, VARSIZE_ANY( vec ) );
      /* ================================= */

      /* Create Neighbor Tuple */
      HnswNeighborTuple ntup = palloc0( ntupSize );
      uint32 slot_count = 0;
      ldb_unaligned_slot_union_t* slots
         = get_node_neighbors_mut( &metadata, node, node_level, &slot_count );
      ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;
      ntup->count = HnswGetLayerM(metadata.connectivity, node_level);
      ntup->unused = 0;
      for( uint32 j = 0; j < ntup->count; j++ ) {
      	 if (j < slot_count) {
           memcpy( &ntup->indextids[ j ].ip_blkid, &slots[ j ].seqid, sizeof( slots[ j ].seqid ) );
         } else {
		   ItemPointerSetInvalid(&ntup->indextids[ j ]);
         }
         elog( INFO, "index: %u, seqid %u written as %u", j, slots[ j ].seqid, ntup->indextids[ j ].ip_blkid );
      }
      elog( INFO, "ntup is %p", ntup );
      /* ================================= */

      if( PageGetFreeSpace( page ) < combinedSize ) {
   		 HnswBuildAppendPage(index, &buf, &page, forkNum);
         blockno = BufferGetBlockNumber( buf );
         elog( INFO, "created new page" );
      }

	  // get the offset number of neighbor tuple we will insert later
	  ItemPointerSet(&etup->neighbortid, blockno, OffsetNumberNext(OffsetNumberNext(offsetno)));

      ItemPointerData tid = { 0 };

      offsetno = PageAddItem(
         page, (Item)etup, etupSize, InvalidOffsetNumber, false, false );

	  ItemPointerSet(&tid, blockno, offsetno);

      if( node_id == entry_slot ) {
         entry_level = node_level;
         memcpy( &entry_tid, &tid, sizeof( ItemPointerData ) );
      }

      memcpy( &item_pointers[ node_id ], &tid, sizeof( ItemPointerData ) );

      offsetno = PageAddItem(
         page, (Item)ntup, ntupSize, InvalidOffsetNumber, false, false );

      progress += node_size;
   }
   UnlockReleaseBuffer( buf );

   BlockNumber last_data_block = blockno;
   // update entry tid
   buf = ReadBufferExtended( index, forkNum, 0, RBM_NORMAL, GetAccessStrategy( BAS_BULKREAD ) );
   LockBuffer( buf, BUFFER_LOCK_EXCLUSIVE );
   page = BufferGetPage( buf );
   HnswMetaPage metap = HnswPageGetMeta( page );

   metap->insertPage = last_data_block;
   metap->entryBlkno = BlockIdGetBlockNumber( &entry_tid.ip_blkid );
   metap->entryOffno = entry_tid.ip_posid;
   metap->entryLevel = entry_level;

   MarkBufferDirty( buf );
   UnlockReleaseBuffer( buf );


   // rewrite neighbors
   for( BlockNumber blockno = 1;
        blockno <= last_data_block;
        blockno++ ) {
      buf = ReadBufferExtended( index, forkNum, blockno, RBM_NORMAL, GetAccessStrategy( BAS_BULKREAD ) );
      LockBuffer( buf, BUFFER_LOCK_EXCLUSIVE );
      page = BufferGetPage( buf );
      OffsetNumber maxoffset = PageGetMaxOffsetNumber( page );

      for( OffsetNumber offset = FirstOffsetNumber; offset <= maxoffset; offset = OffsetNumberNext( offset ) ) {
         HnswNeighborTuple neighborpage = (HnswNeighborTuple)PageGetItem( page, PageGetItemId( page, offset ) );
         if( !HnswIsNeighborTuple( neighborpage ) )
            continue;
         for( uint32 i = 0; i < neighborpage->count; i++ ) {
         	if (!BlockNumberIsValid(BlockIdGetBlockNumber(&neighborpage->indextids[ i ].ip_blkid))) {
         			continue;
         	}

            uint32 seqid = 0;
            memcpy( &seqid, &neighborpage->indextids[ i ].ip_blkid, sizeof( uint32 ) );
            memcpy( &neighborpage->indextids[ i ], &item_pointers[ seqid ], sizeof( ItemPointerData ) );
         }
      }

      MarkBufferDirty( buf );
      UnlockReleaseBuffer( buf );
   }

   close( index_file_fd );
}
