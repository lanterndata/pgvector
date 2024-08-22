#ifndef EXTERNAL_IDX_SOCKET_SSL_H
#define EXTERNAL_IDX_SOCKET_SSL_H

#include <pg_config.h>
#ifdef USE_OPENSSL
#include <openssl/err.h>
#include <openssl/ssl.h>
#else
#define SSL_CTX void
#define SSL void
#endif

#endif // EXTERNAL_IDX_SOCKET_SSL_H
