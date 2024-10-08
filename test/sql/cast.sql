SELECT ARRAY[1,2,3]::vector;
SELECT ARRAY[1.0,2.0,3.0]::vector;
SELECT ARRAY[1,2,3]::float4[]::vector;
SELECT ARRAY[1,2,3]::float8[]::vector;
SELECT ARRAY[1,2,3]::numeric[]::vector;

SELECT '[1,2,3]'::vector::real[];

SELECT '{1,2,3}'::real[]::vector;
SELECT '{1,2,3}'::real[]::vector(3);
SELECT '{1,2,3}'::real[]::vector(2);
SELECT '{NULL}'::real[]::vector;
SELECT '{NaN}'::real[]::vector;
SELECT '{Infinity}'::real[]::vector;
SELECT '{-Infinity}'::real[]::vector;
SELECT '{}'::real[]::vector;
SELECT '{{1}}'::real[]::vector;

SELECT '{1,2,3}'::double precision[]::vector;
SELECT '{1,2,3}'::double precision[]::vector(3);
SELECT '{1,2,3}'::double precision[]::vector(2);
SELECT '{4e38,-4e38}'::double precision[]::vector;
SELECT '{1e-46,-1e-46}'::double precision[]::vector;

SELECT '[1,2,3]'::vector::halfvec;
SELECT '[1,2,3]'::vector::halfvec(3);
SELECT '[1,2,3]'::vector::halfvec(2);
SELECT '[65520]'::vector::halfvec;
SELECT '[1e-8]'::vector::halfvec;

SELECT '[1,2,3]'::halfvec::vector;
SELECT '[1,2,3]'::halfvec::vector(3);
SELECT '[1,2,3]'::halfvec::vector(2);

SELECT '{1,2,3}'::real[]::halfvec;
SELECT '{1,2,3}'::real[]::halfvec(3);
SELECT '{1,2,3}'::real[]::halfvec(2);
SELECT '{65520,-65520}'::real[]::halfvec;
SELECT '{1e-8,-1e-8}'::real[]::halfvec;

SELECT '[0,1.5,0,3.5,0]'::vector::sparsevec;
SELECT '[0,1.5,0,3.5,0]'::vector::sparsevec(5);
SELECT '[0,1.5,0,3.5,0]'::vector::sparsevec(4);

SELECT '{2:1.5,4:3.5}/5'::sparsevec::vector;
SELECT '{2:1.5,4:3.5}/5'::sparsevec::vector(5);
SELECT '{2:1.5,4:3.5}/5'::sparsevec::vector(4);
SELECT '{}/16001'::sparsevec::vector;

SELECT '[0,1.5,0,3.5,0]'::halfvec::sparsevec;
SELECT '[0,1.5,0,3.5,0]'::halfvec::sparsevec(5);
SELECT '[0,1.5,0,3.5,0]'::halfvec::sparsevec(4);

SELECT '{2:1.5,4:3.5}/5'::sparsevec::halfvec;
SELECT '{2:1.5,4:3.5}/5'::sparsevec::halfvec(5);
SELECT '{2:1.5,4:3.5}/5'::sparsevec::halfvec(4);
SELECT '{}/16001'::sparsevec::halfvec;
SELECT '{1:65520}/1'::sparsevec::halfvec;
SELECT '{1:1e-8}/1'::sparsevec::halfvec;

SELECT '{2:1.5,4:3.5}/5'::sparsevec::real[];
SELECT array_length('{}/16001'::sparsevec::real[], 1);

SELECT array_agg(n)::vector FROM generate_series(1, 16001) n;
SELECT array_to_vector(array_agg(n), 16001, false) FROM generate_series(1, 16001) n;

SELECT ARRAY[NULL]::real[]::sparsevec;
SELECT ARRAY[NULL,1,2,3]::sparsevec;
SELECT ARRAY[1,2,3]::sparsevec;
SELECT ARRAY[.1,2,3]::sparsevec;
SELECT ARRAY[1,2,3]::float4[]::sparsevec;
SELECT ARRAY[1,2,3]::float8[]::sparsevec;
SELECT ARRAY[1,2,3]::numeric[]::sparsevec;
SELECT ARRAY[1,2,0,3]::real[]::sparsevec(2);
SELECT ARRAY[1,2,0,3]::real[]::sparsevec(4);
SELECT ARRAY[1,2,0,3]::real[]::sparsevec(40);
SELECT array_agg(n)::sparsevec FROM generate_series(1, 16001) n;
SELECT ARRAY(SELECT (i % 307 = 0)::integer FROM generate_series(1, 40000) AS i)::sparsevec;

-- ensure no error
SELECT ARRAY[1,2,3] = ARRAY[1,2,3];
