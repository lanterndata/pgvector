#ifndef HNSW_USEARCH_STORAGE_H
#define HNSW_USEARCH_STORAGE_H
#ifdef __cplusplus
extern "C" {
#endif
#include "postgres.h"
#include "storage/itemptr.h"
#include "usearch.h"

#define USEARCH_HEADER_SIZE 136
const usearch_label_t INVALID_ELEMENT_LABEL = 0;
typedef union __attribute__( ( __packed__ ) )
{
   ItemPointerData itemPointerData;
   uint32 seqid;
} ldb_unaligned_slot_union_t;

uint32_t UsearchNodeBytes( const metadata_t* metadata, int vector_bytes, int level );
void usearch_init_node(
   metadata_t* meta, char* tape, usearch_key_t key, uint32_t level, uint64_t slot_id, void* vector, size_t vector_len );

uint32_t node_tuple_size( char* node, uint32_t vector_dim, const metadata_t* meta );
uint32_t node_vector_size( char* node, uint32_t vector_dim, const metadata_t* meta );

usearch_label_t label_from_node( char* node );
unsigned long level_from_node( char* node );
void reset_node_label( char* node );

ldb_unaligned_slot_union_t* get_node_neighbors_mut( const metadata_t* meta,
                                                    char* node,
                                                    uint32_t level,
                                                    uint32_t* neighbors_count );

#ifdef __cplusplus
}
#endif

#endif  // HNSW_USEARCH_STORAGE_H
