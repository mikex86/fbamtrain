.macro EMBED_ASSET sym, path, align=4

#if defined(__ELF__)
    .section .rodata.kernels,"a",@progbits
#elif defined(__MACH__)
    .section __TEXT,__const
#else
    .section .rdata
#endif

    .p2align \align

    .globl  \sym\()_start
    .globl  \sym\()_end

#if defined(__ELF__)
    .type   \sym\()_start, @object
    .type   \sym\()_end,   @object
#endif

\sym\()_start:
    .incbin "\path"
\sym\()_end:

#if defined(__ELF__)
    .globl  \sym\()_size
    .set    \sym\()_size, \sym\()_end - \sym\()_start
    .size   \sym\()_start, .-\sym\()_start
#endif

.endm