#if FLOAT_BITS == 64
    typedef double real;
    typedef double2 real2;
    typedef double3 real3;
    typedef double4 real4;
    typedef double8 real8;
    typedef double16 real16;
#elif FLOAT_BITS == 32
    typedef float real;
    typedef float2 real2;
    typedef float3 real3;
    typedef float4 real4;
    typedef float8 real8;
    typedef float16 real16;
#elif FLOAT_BITS == 16
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    typedef half real;
    typedef half2 real2;
    typedef half3 real3;
    typedef half4 real4;
    typedef half8 real8;
    typedef half16 real16;
#endif


#if VECTOR_WIDTH == 1 || VEC_WIDTH == 1
    typedef real realX;
    #define VEC_DOT_SCALAR(v,n) (v * n)
    #define VEC_REDUCE(v,r) (v)
#elif VECTOR_WIDTH == 2 || VEC_WIDTH == 2
    typedef real2 realX;
    #define VEC_DOT_SCALAR(v,n) dot(v, (realX)(n))
    #define VEC_REDUCE(v,r) r(v.s0, v.s1)
#elif VECTOR_WIDTH == 4 || VEC_WIDTH == 4
    typedef real4 realX;
    #define VEC_DOT_SCALAR(v,n) dot(v, (realX)(n))
    #define VEC_REDUCE(v,r) r(r(v.s0, v.s1), r(v.s2, v.s3))
#elif VECTOR_WIDTH == 8 || VEC_WIDTH == 8
    typedef real8 realX;
    #define VEC_DOT_SCALAR(v,n) (dot(v.s0123, (real4)(n)) + dot(v.s4567, (real4)(n)))
    #define VEC_REDUCE(v,r) r(r(r(v.s0, v.s1), r(v.s2, v.s3)), r(r(v.s4, v.s5), r(v.s6, v.s7)))
#elif VECTOR_WIDTH == 16 || VEC_WIDTH == 16
    typedef real16 realX;
    #define VEC_DOT_SCALAR(v,n) (dot(v.s0123, (real4)(n)) + dot(v.s4567, (real4)(n)) + dot(v.s89ab, (real4)(n)) + dot(v.scdef, (real4)(n)))
    #define VEC_REDUCE(v,r) r(r(r(r(v.s0, v.s1), r(v.s2, v.s3)), r(r(v.s4, v.s5), r(v.s6, v.s7))), r(r(r(v.s8, v.s9), r(v.sa, v.sb)), r(r(v.sc, v.sd), r(v.se, v.sf))))
#endif


#if VECTOR_WIDTH == 1 || VEC_WIDTH == 1
    #define VEC_IDX(v,i) v
#else
    #define VEC_IDX(v,i) v[i]
#endif

#define VEC_MAX(v) VEC_REDUCE(v, fmax)
#define VEC_MIN(v) VEC_REDUCE(v, fmin)