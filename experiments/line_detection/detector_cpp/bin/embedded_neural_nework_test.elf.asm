
bin/embedded_neural_nework_test.elf:	file format Mach-O 64-bit x86-64


Disassembly of section __TEXT,__text:

0000000100001c50 __Z8get_timev:
100001c50: 55                          	pushq	%rbp
100001c51: 48 89 e5                    	movq	%rsp, %rbp
100001c54: e8 4f 52 00 00              	callq	21071 <dyld_stub_binder+0x100006ea8>
100001c59: c4 e1 fb 2a c0              	vcvtsi2sd	%rax, %xmm0, %xmm0
100001c5e: c5 fb 5e 05 1a 54 00 00     	vdivsd	21530(%rip), %xmm0, %xmm0
100001c66: 5d                          	popq	%rbp
100001c67: c3                          	retq
100001c68: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100001c70 __Z14get_predictionRN2cv3MatER11LineNetworkf:
100001c70: 55                          	pushq	%rbp
100001c71: 48 89 e5                    	movq	%rsp, %rbp
100001c74: 41 57                       	pushq	%r15
100001c76: 41 56                       	pushq	%r14
100001c78: 41 55                       	pushq	%r13
100001c7a: 41 54                       	pushq	%r12
100001c7c: 53                          	pushq	%rbx
100001c7d: 48 81 ec 18 01 00 00        	subq	$280, %rsp
100001c84: c5 fa 11 45 b0              	vmovss	%xmm0, -80(%rbp)
100001c89: 49 89 d5                    	movq	%rdx, %r13
100001c8c: 49 89 f4                    	movq	%rsi, %r12
100001c8f: 49 89 fe                    	movq	%rdi, %r14
100001c92: 48 8b 05 c7 73 00 00        	movq	29639(%rip), %rax
100001c99: 48 8b 00                    	movq	(%rax), %rax
100001c9c: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100001ca0: 8b 46 08                    	movl	8(%rsi), %eax
100001ca3: 8b 4e 0c                    	movl	12(%rsi), %ecx
100001ca6: c7 85 e0 fe ff ff 00 00 ff 42       	movl	$1124007936, -288(%rbp)
100001cb0: 48 8d 95 e8 fe ff ff        	leaq	-280(%rbp), %rdx
100001cb7: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001cbb: c5 fc 11 85 e4 fe ff ff     	vmovups	%ymm0, -284(%rbp)
100001cc3: c5 fc 11 85 00 ff ff ff     	vmovups	%ymm0, -256(%rbp)
100001ccb: 48 89 95 20 ff ff ff        	movq	%rdx, -224(%rbp)
100001cd2: 48 8d 95 30 ff ff ff        	leaq	-208(%rbp), %rdx
100001cd9: 48 89 95 28 ff ff ff        	movq	%rdx, -216(%rbp)
100001ce0: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001ce4: c5 f8 11 85 30 ff ff ff     	vmovups	%xmm0, -208(%rbp)
100001cec: 89 4d b8                    	movl	%ecx, -72(%rbp)
100001cef: 89 45 bc                    	movl	%eax, -68(%rbp)
100001cf2: 4c 8d bd e0 fe ff ff        	leaq	-288(%rbp), %r15
100001cf9: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
100001cfd: 4c 89 ff                    	movq	%r15, %rdi
100001d00: be 02 00 00 00              	movl	$2, %esi
100001d05: 31 c9                       	xorl	%ecx, %ecx
100001d07: c5 f8 77                    	vzeroupper
100001d0a: e8 2d 51 00 00              	callq	20781 <dyld_stub_binder+0x100006e3c>
100001d0f: 48 c7 85 50 ff ff ff 00 00 00 00    	movq	$0, -176(%rbp)
100001d1a: c7 85 40 ff ff ff 00 00 01 01       	movl	$16842752, -192(%rbp)
100001d24: 4c 89 a5 48 ff ff ff        	movq	%r12, -184(%rbp)
100001d2b: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
100001d33: c7 45 b8 00 00 01 02        	movl	$33619968, -72(%rbp)
100001d3a: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
100001d3e: 48 8d bd 40 ff ff ff        	leaq	-192(%rbp), %rdi
100001d45: 48 8d 75 b8                 	leaq	-72(%rbp), %rsi
100001d49: ba 06 00 00 00              	movl	$6, %edx
100001d4e: 31 c9                       	xorl	%ecx, %ecx
100001d50: e8 11 51 00 00              	callq	20753 <dyld_stub_binder+0x100006e66>
100001d55: 41 8b 44 24 08              	movl	8(%r12), %eax
100001d5a: 41 8b 4c 24 0c              	movl	12(%r12), %ecx
100001d5f: c7 85 40 ff ff ff 00 00 ff 42       	movl	$1124007936, -192(%rbp)
100001d69: 48 8d 95 48 ff ff ff        	leaq	-184(%rbp), %rdx
100001d70: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001d74: c5 fc 11 85 44 ff ff ff     	vmovups	%ymm0, -188(%rbp)
100001d7c: c5 fc 11 85 60 ff ff ff     	vmovups	%ymm0, -160(%rbp)
100001d84: 48 89 55 80                 	movq	%rdx, -128(%rbp)
100001d88: 48 8d 55 90                 	leaq	-112(%rbp), %rdx
100001d8c: 48 89 55 88                 	movq	%rdx, -120(%rbp)
100001d90: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001d94: c5 f8 11 45 90              	vmovups	%xmm0, -112(%rbp)
100001d99: 89 4d b8                    	movl	%ecx, -72(%rbp)
100001d9c: 89 45 bc                    	movl	%eax, -68(%rbp)
100001d9f: 4c 8d a5 40 ff ff ff        	leaq	-192(%rbp), %r12
100001da6: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
100001daa: 4c 89 e7                    	movq	%r12, %rdi
100001dad: be 02 00 00 00              	movl	$2, %esi
100001db2: 31 c9                       	xorl	%ecx, %ecx
100001db4: c5 f8 77                    	vzeroupper
100001db7: e8 80 50 00 00              	callq	20608 <dyld_stub_binder+0x100006e3c>
100001dbc: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
100001dc4: c7 45 b8 00 00 01 01        	movl	$16842752, -72(%rbp)
100001dcb: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
100001dcf: 48 c7 85 d8 fe ff ff 00 00 00 00    	movq	$0, -296(%rbp)
100001dda: c7 85 c8 fe ff ff 00 00 01 02       	movl	$33619968, -312(%rbp)
100001de4: 4c 89 a5 d0 fe ff ff        	movq	%r12, -304(%rbp)
100001deb: 41 8b 45 0c                 	movl	12(%r13), %eax
100001def: 41 8b 4d 10                 	movl	16(%r13), %ecx
100001df3: 89 4d a0                    	movl	%ecx, -96(%rbp)
100001df6: 89 45 a4                    	movl	%eax, -92(%rbp)
100001df9: 48 8d 7d b8                 	leaq	-72(%rbp), %rdi
100001dfd: 48 8d b5 c8 fe ff ff        	leaq	-312(%rbp), %rsi
100001e04: 48 8d 55 a0                 	leaq	-96(%rbp), %rdx
100001e08: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001e0c: c5 f0 57 c9                 	vxorps	%xmm1, %xmm1, %xmm1
100001e10: b9 03 00 00 00              	movl	$3, %ecx
100001e15: e8 3a 50 00 00              	callq	20538 <dyld_stub_binder+0x100006e54>
100001e1a: 41 8b 55 0c                 	movl	12(%r13), %edx
100001e1e: 85 d2                       	testl	%edx, %edx
100001e20: 74 7d                       	je	125 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x22f>
100001e22: 41 8b 75 10                 	movl	16(%r13), %esi
100001e26: 45 31 c0                    	xorl	%r8d, %r8d
100001e29: 45 31 c9                    	xorl	%r9d, %r9d
100001e2c: 85 f6                       	testl	%esi, %esi
100001e2e: 75 0e                       	jne	14 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x1ce>
100001e30: 31 f6                       	xorl	%esi, %esi
100001e32: 41 ff c0                    	incl	%r8d
100001e35: 41 39 d0                    	cmpl	%edx, %r8d
100001e38: 73 65                       	jae	101 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x22f>
100001e3a: 85 f6                       	testl	%esi, %esi
100001e3c: 74 f2                       	je	-14 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x1c0>
100001e3e: 49 63 d0                    	movslq	%r8d, %rdx
100001e41: 31 ff                       	xorl	%edi, %edi
100001e43: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100001e4d: 0f 1f 00                    	nopl	(%rax)
100001e50: 41 8d 34 39                 	leal	(%r9,%rdi), %esi
100001e54: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100001e58: 48 8b 1b                    	movq	(%rbx), %rbx
100001e5b: 48 0f af da                 	imulq	%rdx, %rbx
100001e5f: 48 03 9d 50 ff ff ff        	addq	-176(%rbp), %rbx
100001e66: 48 63 ff                    	movslq	%edi, %rdi
100001e69: 0f b6 1c 1f                 	movzbl	(%rdi,%rbx), %ebx
100001e6d: d0 eb                       	shrb	%bl
100001e6f: 49 8d 45 28                 	leaq	40(%r13), %rax
100001e73: 49 8d 4d 30                 	leaq	48(%r13), %rcx
100001e77: 41 80 7d 24 00              	cmpb	$0, 36(%r13)
100001e7c: 48 0f 44 c8                 	cmoveq	%rax, %rcx
100001e80: 48 8b 01                    	movq	(%rcx), %rax
100001e83: 88 1c 30                    	movb	%bl, (%rax,%rsi)
100001e86: ff c7                       	incl	%edi
100001e88: 41 8b 75 10                 	movl	16(%r13), %esi
100001e8c: 39 f7                       	cmpl	%esi, %edi
100001e8e: 72 c0                       	jb	-64 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x1e0>
100001e90: 41 8b 55 0c                 	movl	12(%r13), %edx
100001e94: 41 01 f9                    	addl	%edi, %r9d
100001e97: 41 ff c0                    	incl	%r8d
100001e9a: 41 39 d0                    	cmpl	%edx, %r8d
100001e9d: 72 9b                       	jb	-101 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x1ca>
100001e9f: 49 8b 45 00                 	movq	(%r13), %rax
100001ea3: 4c 89 ef                    	movq	%r13, %rdi
100001ea6: ff 50 10                    	callq	*16(%rax)
100001ea9: 41 8b 45 18                 	movl	24(%r13), %eax
100001ead: 41 8b 4d 1c                 	movl	28(%r13), %ecx
100001eb1: 41 c7 06 00 00 ff 42        	movl	$1124007936, (%r14)
100001eb8: 49 8d 56 08                 	leaq	8(%r14), %rdx
100001ebc: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001ec0: c4 c1 7c 11 46 04           	vmovups	%ymm0, 4(%r14)
100001ec6: c4 c1 7c 11 46 20           	vmovups	%ymm0, 32(%r14)
100001ecc: 49 89 56 40                 	movq	%rdx, 64(%r14)
100001ed0: 49 8d 56 50                 	leaq	80(%r14), %rdx
100001ed4: 49 89 56 48                 	movq	%rdx, 72(%r14)
100001ed8: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001edc: c4 c1 78 11 46 50           	vmovups	%xmm0, 80(%r14)
100001ee2: 89 4d b8                    	movl	%ecx, -72(%rbp)
100001ee5: 89 45 bc                    	movl	%eax, -68(%rbp)
100001ee8: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
100001eec: 4c 89 f7                    	movq	%r14, %rdi
100001eef: be 02 00 00 00              	movl	$2, %esi
100001ef4: 31 c9                       	xorl	%ecx, %ecx
100001ef6: c5 f8 77                    	vzeroupper
100001ef9: e8 3e 4f 00 00              	callq	20286 <dyld_stub_binder+0x100006e3c>
100001efe: 41 8b 45 18                 	movl	24(%r13), %eax
100001f02: 41 83 7d 14 01              	cmpl	$1, 20(%r13)
100001f07: 0f 85 d6 00 00 00           	jne	214 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x373>
100001f0d: 85 c0                       	testl	%eax, %eax
100001f0f: 0f 84 d7 02 00 00           	je	727 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001f15: c5 fa 10 45 b0              	vmovss	-80(%rbp), %xmm0
100001f1a: c5 fa 59 05 a6 51 00 00     	vmulss	20902(%rip), %xmm0, %xmm0
100001f22: 41 8b 7d 1c                 	movl	28(%r13), %edi
100001f26: 45 31 c0                    	xorl	%r8d, %r8d
100001f29: 31 f6                       	xorl	%esi, %esi
100001f2b: 85 ff                       	testl	%edi, %edi
100001f2d: 75 27                       	jne	39 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2e6>
100001f2f: e9 9c 00 00 00              	jmp	156 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x360>
100001f34: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100001f3e: 66 90                       	nop
100001f40: 41 8b 45 18                 	movl	24(%r13), %eax
100001f44: 01 d6                       	addl	%edx, %esi
100001f46: 41 ff c0                    	incl	%r8d
100001f49: 41 39 c0                    	cmpl	%eax, %r8d
100001f4c: 0f 83 9a 02 00 00           	jae	666 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001f52: 85 ff                       	testl	%edi, %edi
100001f54: 74 7a                       	je	122 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x360>
100001f56: 49 63 c0                    	movslq	%r8d, %rax
100001f59: 31 d2                       	xorl	%edx, %edx
100001f5b: eb 41                       	jmp	65 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x32e>
100001f5d: 0f 1f 00                    	nopl	(%rax)
100001f60: 40 0f be cf                 	movsbl	%dil, %ecx
100001f64: c5 ea 2a c9                 	vcvtsi2ss	%ecx, %xmm2, %xmm1
100001f68: 49 8b 7e 48                 	movq	72(%r14), %rdi
100001f6c: 48 8b 3f                    	movq	(%rdi), %rdi
100001f6f: 48 0f af f8                 	imulq	%rax, %rdi
100001f73: 49 03 7e 10                 	addq	16(%r14), %rdi
100001f77: 48 63 d2                    	movslq	%edx, %rdx
100001f7a: 88 0c 3a                    	movb	%cl, (%rdx,%rdi)
100001f7d: 49 8b 4e 48                 	movq	72(%r14), %rcx
100001f81: 48 8b 09                    	movq	(%rcx), %rcx
100001f84: 48 0f af c8                 	imulq	%rax, %rcx
100001f88: 49 03 4e 10                 	addq	16(%r14), %rcx
100001f8c: c5 f8 2e c8                 	vucomiss	%xmm0, %xmm1
100001f90: 0f 97 04 0a                 	seta	(%rdx,%rcx)
100001f94: ff c2                       	incl	%edx
100001f96: 41 8b 7d 1c                 	movl	28(%r13), %edi
100001f9a: 39 fa                       	cmpl	%edi, %edx
100001f9c: 73 a2                       	jae	-94 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2d0>
100001f9e: 8d 3c 16                    	leal	(%rsi,%rdx), %edi
100001fa1: 49 8d 5d 30                 	leaq	48(%r13), %rbx
100001fa5: 49 8d 4d 28                 	leaq	40(%r13), %rcx
100001fa9: 41 80 7d 24 00              	cmpb	$0, 36(%r13)
100001fae: 48 0f 44 cb                 	cmoveq	%rbx, %rcx
100001fb2: 48 8b 09                    	movq	(%rcx), %rcx
100001fb5: 0f b6 3c 39                 	movzbl	(%rcx,%rdi), %edi
100001fb9: 40 84 ff                    	testb	%dil, %dil
100001fbc: 79 a2                       	jns	-94 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2f0>
100001fbe: 31 ff                       	xorl	%edi, %edi
100001fc0: eb 9e                       	jmp	-98 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2f0>
100001fc2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100001fcc: 0f 1f 40 00                 	nopl	(%rax)
100001fd0: 31 ff                       	xorl	%edi, %edi
100001fd2: 41 ff c0                    	incl	%r8d
100001fd5: 41 39 c0                    	cmpl	%eax, %r8d
100001fd8: 0f 82 74 ff ff ff           	jb	-140 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2e2>
100001fde: e9 09 02 00 00              	jmp	521 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001fe3: 85 c0                       	testl	%eax, %eax
100001fe5: 0f 84 01 02 00 00           	je	513 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001feb: c5 fa 10 45 b0              	vmovss	-80(%rbp), %xmm0
100001ff0: c5 fa 59 05 d0 50 00 00     	vmulss	20688(%rip), %xmm0, %xmm0
100001ff8: 41 8b 4d 1c                 	movl	28(%r13), %ecx
100001ffc: 45 31 c0                    	xorl	%r8d, %r8d
100001fff: 31 f6                       	xorl	%esi, %esi
100002001: 45 31 d2                    	xorl	%r10d, %r10d
100002004: 85 c9                       	testl	%ecx, %ecx
100002006: 75 21                       	jne	33 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3b9>
100002008: e9 d3 01 00 00              	jmp	467 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x570>
10000200d: 0f 1f 00                    	nopl	(%rax)
100002010: 41 8b 45 18                 	movl	24(%r13), %eax
100002014: 8b 75 ac                    	movl	-84(%rbp), %esi
100002017: ff c6                       	incl	%esi
100002019: 39 c6                       	cmpl	%eax, %esi
10000201b: 0f 83 cb 01 00 00           	jae	459 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100002021: 85 c9                       	testl	%ecx, %ecx
100002023: 0f 84 b7 01 00 00           	je	439 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x570>
100002029: 89 75 ac                    	movl	%esi, -84(%rbp)
10000202c: 48 63 d6                    	movslq	%esi, %rdx
10000202f: 45 31 db                    	xorl	%r11d, %r11d
100002032: 48 89 55 b0                 	movq	%rdx, -80(%rbp)
100002036: 45 8b 7d 14                 	movl	20(%r13), %r15d
10000203a: 45 85 ff                    	testl	%r15d, %r15d
10000203d: 75 3f                       	jne	63 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x40e>
10000203f: 90                          	nop
100002040: b0 81                       	movb	$-127, %al
100002042: 31 ff                       	xorl	%edi, %edi
100002044: 0f be c0                    	movsbl	%al, %eax
100002047: c5 ea 2a c8                 	vcvtsi2ss	%eax, %xmm2, %xmm1
10000204b: c5 f8 2e c8                 	vucomiss	%xmm0, %xmm1
10000204f: 41 0f 46 f8                 	cmovbel	%r8d, %edi
100002053: 49 8b 46 48                 	movq	72(%r14), %rax
100002057: 48 8b 00                    	movq	(%rax), %rax
10000205a: 48 0f af c2                 	imulq	%rdx, %rax
10000205e: 49 03 46 10                 	addq	16(%r14), %rax
100002062: 4d 63 db                    	movslq	%r11d, %r11
100002065: 41 88 3c 03                 	movb	%dil, (%r11,%rax)
100002069: 41 ff c3                    	incl	%r11d
10000206c: 41 8b 4d 1c                 	movl	28(%r13), %ecx
100002070: 41 39 cb                    	cmpl	%ecx, %r11d
100002073: 73 9b                       	jae	-101 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3a0>
100002075: 45 8b 7d 14                 	movl	20(%r13), %r15d
100002079: 45 85 ff                    	testl	%r15d, %r15d
10000207c: 74 c2                       	je	-62 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3d0>
10000207e: 49 8d 45 30                 	leaq	48(%r13), %rax
100002082: 41 80 7d 24 00              	cmpb	$0, 36(%r13)
100002087: 49 8d 4d 28                 	leaq	40(%r13), %rcx
10000208b: 48 0f 44 c8                 	cmoveq	%rax, %rcx
10000208f: 4c 8b 09                    	movq	(%rcx), %r9
100002092: 41 8d 47 ff                 	leal	-1(%r15), %eax
100002096: 45 89 fc                    	movl	%r15d, %r12d
100002099: 41 83 e4 07                 	andl	$7, %r12d
10000209d: 83 f8 07                    	cmpl	$7, %eax
1000020a0: 73 1e                       	jae	30 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x450>
1000020a2: b0 81                       	movb	$-127, %al
1000020a4: 31 db                       	xorl	%ebx, %ebx
1000020a6: 31 ff                       	xorl	%edi, %edi
1000020a8: 45 85 e4                    	testl	%r12d, %r12d
1000020ab: 0f 85 ff 00 00 00           	jne	255 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x540>
1000020b1: eb 91                       	jmp	-111 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3d4>
1000020b3: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000020bd: 0f 1f 00                    	nopl	(%rax)
1000020c0: 45 29 e7                    	subl	%r12d, %r15d
1000020c3: b0 81                       	movb	$-127, %al
1000020c5: 31 db                       	xorl	%ebx, %ebx
1000020c7: 31 ff                       	xorl	%edi, %edi
1000020c9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
1000020d0: 41 8d 34 1a                 	leal	(%r10,%rbx), %esi
1000020d4: 41 0f b6 34 31              	movzbl	(%r9,%rsi), %esi
1000020d9: 45 8d 04 1a                 	leal	(%r10,%rbx), %r8d
1000020dd: 41 83 c0 01                 	addl	$1, %r8d
1000020e1: 40 38 c6                    	cmpb	%al, %sil
1000020e4: 0f 4f fb                    	cmovgl	%ebx, %edi
1000020e7: 0f b6 c0                    	movzbl	%al, %eax
1000020ea: 0f 4d c6                    	cmovgel	%esi, %eax
1000020ed: 43 0f b6 34 01              	movzbl	(%r9,%r8), %esi
1000020f2: 41 8d 0c 1a                 	leal	(%r10,%rbx), %ecx
1000020f6: 83 c1 02                    	addl	$2, %ecx
1000020f9: 8d 53 01                    	leal	1(%rbx), %edx
1000020fc: 40 38 c6                    	cmpb	%al, %sil
1000020ff: 0f 4e d7                    	cmovlel	%edi, %edx
100002102: 0f 4d c6                    	cmovgel	%esi, %eax
100002105: 41 0f b6 0c 09              	movzbl	(%r9,%rcx), %ecx
10000210a: 41 8d 34 1a                 	leal	(%r10,%rbx), %esi
10000210e: 83 c6 03                    	addl	$3, %esi
100002111: 8d 7b 02                    	leal	2(%rbx), %edi
100002114: 38 c1                       	cmpb	%al, %cl
100002116: 0f 4e fa                    	cmovlel	%edx, %edi
100002119: 0f 4d c1                    	cmovgel	%ecx, %eax
10000211c: 41 0f b6 0c 31              	movzbl	(%r9,%rsi), %ecx
100002121: 41 8d 14 1a                 	leal	(%r10,%rbx), %edx
100002125: 83 c2 04                    	addl	$4, %edx
100002128: 8d 73 03                    	leal	3(%rbx), %esi
10000212b: 38 c1                       	cmpb	%al, %cl
10000212d: 0f 4e f7                    	cmovlel	%edi, %esi
100002130: 0f 4d c1                    	cmovgel	%ecx, %eax
100002133: 41 0f b6 0c 11              	movzbl	(%r9,%rdx), %ecx
100002138: 41 8d 14 1a                 	leal	(%r10,%rbx), %edx
10000213c: 83 c2 05                    	addl	$5, %edx
10000213f: 8d 7b 04                    	leal	4(%rbx), %edi
100002142: 38 c1                       	cmpb	%al, %cl
100002144: 0f 4e fe                    	cmovlel	%esi, %edi
100002147: 0f 4d c1                    	cmovgel	%ecx, %eax
10000214a: 41 0f b6 0c 11              	movzbl	(%r9,%rdx), %ecx
10000214f: 41 8d 14 1a                 	leal	(%r10,%rbx), %edx
100002153: 83 c2 06                    	addl	$6, %edx
100002156: 8d 73 05                    	leal	5(%rbx), %esi
100002159: 38 c1                       	cmpb	%al, %cl
10000215b: 0f 4e f7                    	cmovlel	%edi, %esi
10000215e: 0f 4d c1                    	cmovgel	%ecx, %eax
100002161: 41 0f b6 0c 11              	movzbl	(%r9,%rdx), %ecx
100002166: 41 8d 3c 1a                 	leal	(%r10,%rbx), %edi
10000216a: 83 c7 07                    	addl	$7, %edi
10000216d: 8d 53 06                    	leal	6(%rbx), %edx
100002170: 38 c1                       	cmpb	%al, %cl
100002172: 0f 4e d6                    	cmovlel	%esi, %edx
100002175: 0f 4d c1                    	cmovgel	%ecx, %eax
100002178: 41 0f b6 0c 39              	movzbl	(%r9,%rdi), %ecx
10000217d: 8d 7b 07                    	leal	7(%rbx), %edi
100002180: 38 c1                       	cmpb	%al, %cl
100002182: 0f 4e fa                    	cmovlel	%edx, %edi
100002185: 0f 4d c1                    	cmovgel	%ecx, %eax
100002188: 83 c3 08                    	addl	$8, %ebx
10000218b: 41 39 df                    	cmpl	%ebx, %r15d
10000218e: 0f 85 3c ff ff ff           	jne	-196 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x460>
100002194: 41 01 da                    	addl	%ebx, %r10d
100002197: 45 31 c0                    	xorl	%r8d, %r8d
10000219a: 48 8b 55 b0                 	movq	-80(%rbp), %rdx
10000219e: 45 85 e4                    	testl	%r12d, %r12d
1000021a1: 0f 84 9d fe ff ff           	je	-355 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3d4>
1000021a7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
1000021b0: 44 89 d6                    	movl	%r10d, %esi
1000021b3: 41 0f b6 34 31              	movzbl	(%r9,%rsi), %esi
1000021b8: 41 ff c2                    	incl	%r10d
1000021bb: 40 38 c6                    	cmpb	%al, %sil
1000021be: 0f 4f fb                    	cmovgl	%ebx, %edi
1000021c1: 0f b6 c0                    	movzbl	%al, %eax
1000021c4: 0f 4d c6                    	cmovgel	%esi, %eax
1000021c7: ff c3                       	incl	%ebx
1000021c9: 41 ff cc                    	decl	%r12d
1000021cc: 75 e2                       	jne	-30 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x540>
1000021ce: e9 71 fe ff ff              	jmp	-399 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3d4>
1000021d3: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000021dd: 0f 1f 00                    	nopl	(%rax)
1000021e0: 31 c9                       	xorl	%ecx, %ecx
1000021e2: ff c6                       	incl	%esi
1000021e4: 39 c6                       	cmpl	%eax, %esi
1000021e6: 0f 82 35 fe ff ff           	jb	-459 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3b1>
1000021ec: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
1000021f3: 48 85 c0                    	testq	%rax, %rax
1000021f6: 74 12                       	je	18 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x59a>
1000021f8: f0                          	lock
1000021f9: ff 48 14                    	decl	20(%rax)
1000021fc: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x59a>
1000021fe: 48 8d bd 40 ff ff ff        	leaq	-192(%rbp), %rdi
100002205: e8 2c 4c 00 00              	callq	19500 <dyld_stub_binder+0x100006e36>
10000220a: 48 c7 85 78 ff ff ff 00 00 00 00    	movq	$0, -136(%rbp)
100002215: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002219: c5 fc 11 85 50 ff ff ff     	vmovups	%ymm0, -176(%rbp)
100002221: 83 bd 44 ff ff ff 00        	cmpl	$0, -188(%rbp)
100002228: 7e 1c                       	jle	28 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x5d6>
10000222a: 48 8b 45 80                 	movq	-128(%rbp), %rax
10000222e: 31 c9                       	xorl	%ecx, %ecx
100002230: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002237: 48 ff c1                    	incq	%rcx
10000223a: 48 63 95 44 ff ff ff        	movslq	-188(%rbp), %rdx
100002241: 48 39 d1                    	cmpq	%rdx, %rcx
100002244: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x5c0>
100002246: 48 8b 7d 88                 	movq	-120(%rbp), %rdi
10000224a: 48 8d 45 90                 	leaq	-112(%rbp), %rax
10000224e: 48 39 c7                    	cmpq	%rax, %rdi
100002251: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x5eb>
100002253: c5 f8 77                    	vzeroupper
100002256: e8 11 4c 00 00              	callq	19473 <dyld_stub_binder+0x100006e6c>
10000225b: 48 8b 85 18 ff ff ff        	movq	-232(%rbp), %rax
100002262: 48 85 c0                    	testq	%rax, %rax
100002265: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x60c>
100002267: f0                          	lock
100002268: ff 48 14                    	decl	20(%rax)
10000226b: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x60c>
10000226d: 48 8d bd e0 fe ff ff        	leaq	-288(%rbp), %rdi
100002274: c5 f8 77                    	vzeroupper
100002277: e8 ba 4b 00 00              	callq	19386 <dyld_stub_binder+0x100006e36>
10000227c: 48 c7 85 18 ff ff ff 00 00 00 00    	movq	$0, -232(%rbp)
100002287: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000228b: c5 fc 11 85 f0 fe ff ff     	vmovups	%ymm0, -272(%rbp)
100002293: 83 bd e4 fe ff ff 00        	cmpl	$0, -284(%rbp)
10000229a: 7e 2a                       	jle	42 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x656>
10000229c: 48 8b 85 20 ff ff ff        	movq	-224(%rbp), %rax
1000022a3: 31 c9                       	xorl	%ecx, %ecx
1000022a5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000022af: 90                          	nop
1000022b0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000022b7: 48 ff c1                    	incq	%rcx
1000022ba: 48 63 95 e4 fe ff ff        	movslq	-284(%rbp), %rdx
1000022c1: 48 39 d1                    	cmpq	%rdx, %rcx
1000022c4: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x640>
1000022c6: 48 8b bd 28 ff ff ff        	movq	-216(%rbp), %rdi
1000022cd: 48 8d 85 30 ff ff ff        	leaq	-208(%rbp), %rax
1000022d4: 48 39 c7                    	cmpq	%rax, %rdi
1000022d7: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x671>
1000022d9: c5 f8 77                    	vzeroupper
1000022dc: e8 8b 4b 00 00              	callq	19339 <dyld_stub_binder+0x100006e6c>
1000022e1: 48 8b 05 78 6d 00 00        	movq	28024(%rip), %rax
1000022e8: 48 8b 00                    	movq	(%rax), %rax
1000022eb: 48 3b 45 d0                 	cmpq	-48(%rbp), %rax
1000022ef: 75 18                       	jne	24 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x699>
1000022f1: 4c 89 f0                    	movq	%r14, %rax
1000022f4: 48 81 c4 18 01 00 00        	addq	$280, %rsp
1000022fb: 5b                          	popq	%rbx
1000022fc: 41 5c                       	popq	%r12
1000022fe: 41 5d                       	popq	%r13
100002300: 41 5e                       	popq	%r14
100002302: 41 5f                       	popq	%r15
100002304: 5d                          	popq	%rbp
100002305: c5 f8 77                    	vzeroupper
100002308: c3                          	retq
100002309: c5 f8 77                    	vzeroupper
10000230c: e8 df 4b 00 00              	callq	19423 <dyld_stub_binder+0x100006ef0>
100002311: 48 89 c7                    	movq	%rax, %rdi
100002314: e8 f7 16 00 00              	callq	5879 <_main+0x15b0>
100002319: 48 89 c7                    	movq	%rax, %rdi
10000231c: e8 ef 16 00 00              	callq	5871 <_main+0x15b0>
100002321: 48 89 c3                    	movq	%rax, %rbx
100002324: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
10000232b: 48 85 c0                    	testq	%rax, %rax
10000232e: 75 2b                       	jne	43 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6eb>
100002330: eb 3b                       	jmp	59 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6fd>
100002332: eb 00                       	jmp	0 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6c4>
100002334: 48 89 c3                    	movq	%rax, %rbx
100002337: 48 8b 85 18 ff ff ff        	movq	-232(%rbp), %rax
10000233e: 48 85 c0                    	testq	%rax, %rax
100002341: 0f 85 83 00 00 00           	jne	131 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x75a>
100002347: e9 93 00 00 00              	jmp	147 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x76f>
10000234c: 48 89 c3                    	movq	%rax, %rbx
10000234f: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
100002356: 48 85 c0                    	testq	%rax, %rax
100002359: 74 12                       	je	18 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6fd>
10000235b: f0                          	lock
10000235c: ff 48 14                    	decl	20(%rax)
10000235f: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6fd>
100002361: 48 8d bd 40 ff ff ff        	leaq	-192(%rbp), %rdi
100002368: e8 c9 4a 00 00              	callq	19145 <dyld_stub_binder+0x100006e36>
10000236d: 48 c7 85 78 ff ff ff 00 00 00 00    	movq	$0, -136(%rbp)
100002378: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000237c: c5 fc 11 85 50 ff ff ff     	vmovups	%ymm0, -176(%rbp)
100002384: 83 bd 44 ff ff ff 00        	cmpl	$0, -188(%rbp)
10000238b: 7e 1c                       	jle	28 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x739>
10000238d: 48 8b 45 80                 	movq	-128(%rbp), %rax
100002391: 31 c9                       	xorl	%ecx, %ecx
100002393: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
10000239a: 48 ff c1                    	incq	%rcx
10000239d: 48 63 95 44 ff ff ff        	movslq	-188(%rbp), %rdx
1000023a4: 48 39 d1                    	cmpq	%rdx, %rcx
1000023a7: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x723>
1000023a9: 48 8b 7d 88                 	movq	-120(%rbp), %rdi
1000023ad: 48 8d 45 90                 	leaq	-112(%rbp), %rax
1000023b1: 48 39 c7                    	cmpq	%rax, %rdi
1000023b4: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x74e>
1000023b6: c5 f8 77                    	vzeroupper
1000023b9: e8 ae 4a 00 00              	callq	19118 <dyld_stub_binder+0x100006e6c>
1000023be: 48 8b 85 18 ff ff ff        	movq	-232(%rbp), %rax
1000023c5: 48 85 c0                    	testq	%rax, %rax
1000023c8: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x76f>
1000023ca: f0                          	lock
1000023cb: ff 48 14                    	decl	20(%rax)
1000023ce: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x76f>
1000023d0: 48 8d bd e0 fe ff ff        	leaq	-288(%rbp), %rdi
1000023d7: c5 f8 77                    	vzeroupper
1000023da: e8 57 4a 00 00              	callq	19031 <dyld_stub_binder+0x100006e36>
1000023df: 48 c7 85 18 ff ff ff 00 00 00 00    	movq	$0, -232(%rbp)
1000023ea: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000023ee: c5 fc 11 85 f0 fe ff ff     	vmovups	%ymm0, -272(%rbp)
1000023f6: 83 bd e4 fe ff ff 00        	cmpl	$0, -284(%rbp)
1000023fd: 7e 27                       	jle	39 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x7b6>
1000023ff: 48 8b 85 20 ff ff ff        	movq	-224(%rbp), %rax
100002406: 31 c9                       	xorl	%ecx, %ecx
100002408: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002410: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002417: 48 ff c1                    	incq	%rcx
10000241a: 48 63 95 e4 fe ff ff        	movslq	-284(%rbp), %rdx
100002421: 48 39 d1                    	cmpq	%rdx, %rcx
100002424: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x7a0>
100002426: 48 8b bd 28 ff ff ff        	movq	-216(%rbp), %rdi
10000242d: 48 8d 85 30 ff ff ff        	leaq	-208(%rbp), %rax
100002434: 48 39 c7                    	cmpq	%rax, %rdi
100002437: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x7d1>
100002439: c5 f8 77                    	vzeroupper
10000243c: e8 2b 4a 00 00              	callq	18987 <dyld_stub_binder+0x100006e6c>
100002441: 48 89 df                    	movq	%rbx, %rdi
100002444: c5 f8 77                    	vzeroupper
100002447: e8 cc 49 00 00              	callq	18892 <dyld_stub_binder+0x100006e18>
10000244c: 0f 0b                       	ud2
10000244e: 48 89 c7                    	movq	%rax, %rdi
100002451: e8 ba 15 00 00              	callq	5562 <_main+0x15b0>
100002456: 48 89 c7                    	movq	%rax, %rdi
100002459: e8 b2 15 00 00              	callq	5554 <_main+0x15b0>
10000245e: 66 90                       	nop

0000000100002460 _main:
100002460: 55                          	pushq	%rbp
100002461: 48 89 e5                    	movq	%rsp, %rbp
100002464: 41 57                       	pushq	%r15
100002466: 41 56                       	pushq	%r14
100002468: 41 55                       	pushq	%r13
10000246a: 41 54                       	pushq	%r12
10000246c: 53                          	pushq	%rbx
10000246d: 48 83 e4 e0                 	andq	$-32, %rsp
100002471: 48 81 ec 00 04 00 00        	subq	$1024, %rsp
100002478: 48 8b 05 e1 6b 00 00        	movq	27617(%rip), %rax
10000247f: 48 8b 00                    	movq	(%rax), %rax
100002482: 48 89 84 24 e0 03 00 00     	movq	%rax, 992(%rsp)
10000248a: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100002492: e8 49 1d 00 00              	callq	7497 <__ZN11LineNetworkC1Ev>
100002497: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000249b: c5 f9 7f 84 24 60 02 00 00  	vmovdqa	%xmm0, 608(%rsp)
1000024a4: 48 c7 84 24 70 02 00 00 00 00 00 00 	movq	$0, 624(%rsp)
1000024b0: bf 30 00 00 00              	movl	$48, %edi
1000024b5: e8 24 4a 00 00              	callq	18980 <dyld_stub_binder+0x100006ede>
1000024ba: 48 89 84 24 70 02 00 00     	movq	%rax, 624(%rsp)
1000024c2: c5 f8 28 05 26 4c 00 00     	vmovaps	19494(%rip), %xmm0
1000024ca: c5 f8 29 84 24 60 02 00 00  	vmovaps	%xmm0, 608(%rsp)
1000024d3: c5 fe 6f 05 01 6a 00 00     	vmovdqu	27137(%rip), %ymm0
1000024db: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
1000024df: 48 b9 69 64 65 6f 2e 6d 70 34       	movabsq	$3778640133568685161, %rcx
1000024e9: 48 89 48 20                 	movq	%rcx, 32(%rax)
1000024ed: c6 40 28 00                 	movb	$0, 40(%rax)
1000024f1: 48 8d bc 24 10 02 00 00     	leaq	528(%rsp), %rdi
1000024f9: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
100002501: 31 d2                       	xorl	%edx, %edx
100002503: c5 f8 77                    	vzeroupper
100002506: e8 19 49 00 00              	callq	18713 <dyld_stub_binder+0x100006e24>
10000250b: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100002513: 74 0d                       	je	13 <_main+0xc2>
100002515: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
10000251d: e8 b0 49 00 00              	callq	18864 <dyld_stub_binder+0x100006ed2>
100002522: 4c 8d 6c 24 68              	leaq	104(%rsp), %r13
100002527: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000252b: c5 f9 d6 44 24 78           	vmovq	%xmm0, 120(%rsp)
100002531: 48 8d 9c 24 10 02 00 00     	leaq	528(%rsp), %rbx
100002539: 4c 8d b4 24 c0 03 00 00     	leaq	960(%rsp), %r14
100002541: 4c 8d a4 24 c0 01 00 00     	leaq	448(%rsp), %r12
100002549: eb 0e                       	jmp	14 <_main+0xf9>
10000254b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100002550: 45 85 ff                    	testl	%r15d, %r15d
100002553: 0f 85 d1 0f 00 00           	jne	4049 <_main+0x10ca>
100002559: 48 89 df                    	movq	%rbx, %rdi
10000255c: c5 f8 77                    	vzeroupper
10000255f: e8 14 49 00 00              	callq	18708 <dyld_stub_binder+0x100006e78>
100002564: 84 c0                       	testb	%al, %al
100002566: 0f 84 be 0f 00 00           	je	4030 <_main+0x10ca>
10000256c: c7 44 24 18 00 00 ff 42     	movl	$1124007936, 24(%rsp)
100002574: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002578: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
10000257d: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
100002582: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002586: 48 8d 44 24 20              	leaq	32(%rsp), %rax
10000258b: 48 89 44 24 58              	movq	%rax, 88(%rsp)
100002590: 4c 89 6c 24 60              	movq	%r13, 96(%rsp)
100002595: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002599: c4 c1 7a 7f 45 00           	vmovdqu	%xmm0, (%r13)
10000259f: 48 89 df                    	movq	%rbx, %rdi
1000025a2: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
1000025a7: c5 f8 77                    	vzeroupper
1000025aa: e8 81 48 00 00              	callq	18561 <dyld_stub_binder+0x100006e30>
1000025af: 41 bf 03 00 00 00           	movl	$3, %r15d
1000025b5: 48 83 7c 24 28 00           	cmpq	$0, 40(%rsp)
1000025bb: 0f 84 8f 08 00 00           	je	2191 <_main+0x9f0>
1000025c1: 8b 44 24 1c                 	movl	28(%rsp), %eax
1000025c5: 83 f8 03                    	cmpl	$3, %eax
1000025c8: 0f 8d 62 03 00 00           	jge	866 <_main+0x4d0>
1000025ce: 48 63 4c 24 20              	movslq	32(%rsp), %rcx
1000025d3: 48 63 74 24 24              	movslq	36(%rsp), %rsi
1000025d8: 48 0f af f1                 	imulq	%rcx, %rsi
1000025dc: 85 c0                       	testl	%eax, %eax
1000025de: 0f 84 6c 08 00 00           	je	2156 <_main+0x9f0>
1000025e4: 48 85 f6                    	testq	%rsi, %rsi
1000025e7: 0f 84 63 08 00 00           	je	2147 <_main+0x9f0>
1000025ed: bf 19 00 00 00              	movl	$25, %edi
1000025f2: c5 f8 77                    	vzeroupper
1000025f5: e8 66 48 00 00              	callq	18534 <dyld_stub_binder+0x100006e60>
1000025fa: 3c 1b                       	cmpb	$27, %al
1000025fc: 0f 84 4e 08 00 00           	je	2126 <_main+0x9f0>
100002602: e8 a1 48 00 00              	callq	18593 <dyld_stub_binder+0x100006ea8>
100002607: 49 89 c5                    	movq	%rax, %r13
10000260a: 48 8d 9c 24 e0 00 00 00     	leaq	224(%rsp), %rbx
100002612: 48 89 df                    	movq	%rbx, %rdi
100002615: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
10000261a: 48 8d 94 24 d8 01 00 00     	leaq	472(%rsp), %rdx
100002622: c5 f9 6e 05 a2 4a 00 00     	vmovd	19106(%rip), %xmm0
10000262a: e8 41 f6 ff ff              	callq	-2495 <__Z14get_predictionRN2cv3MatER11LineNetworkf>
10000262f: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100002637: c5 fa 7e 05 51 4a 00 00     	vmovq	19025(%rip), %xmm0
10000263f: 48 89 de                    	movq	%rbx, %rsi
100002642: e8 2b 48 00 00              	callq	18475 <dyld_stub_binder+0x100006e72>
100002647: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
10000264f: 48 85 c0                    	testq	%rax, %rax
100002652: 74 0e                       	je	14 <_main+0x202>
100002654: f0                          	lock
100002655: ff 48 14                    	decl	20(%rax)
100002658: 75 08                       	jne	8 <_main+0x202>
10000265a: 48 89 df                    	movq	%rbx, %rdi
10000265d: e8 d4 47 00 00              	callq	18388 <dyld_stub_binder+0x100006e36>
100002662: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
10000266e: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100002676: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000267a: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
10000267e: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100002686: 7e 2f                       	jle	47 <_main+0x257>
100002688: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100002690: 31 c9                       	xorl	%ecx, %ecx
100002692: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000269c: 0f 1f 40 00                 	nopl	(%rax)
1000026a0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000026a7: 48 ff c1                    	incq	%rcx
1000026aa: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
1000026b2: 48 39 d1                    	cmpq	%rdx, %rcx
1000026b5: 7c e9                       	jl	-23 <_main+0x240>
1000026b7: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
1000026bf: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
1000026c7: 48 39 c7                    	cmpq	%rax, %rdi
1000026ca: 74 08                       	je	8 <_main+0x274>
1000026cc: c5 f8 77                    	vzeroupper
1000026cf: e8 98 47 00 00              	callq	18328 <dyld_stub_binder+0x100006e6c>
1000026d4: c5 f8 77                    	vzeroupper
1000026d7: e8 cc 47 00 00              	callq	18380 <dyld_stub_binder+0x100006ea8>
1000026dc: 49 89 c7                    	movq	%rax, %r15
1000026df: c7 84 24 e0 00 00 00 00 00 ff 42    	movl	$1124007936, 224(%rsp)
1000026ea: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
1000026f2: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000026f6: c5 fe 7f 40 f4              	vmovdqu	%ymm0, -12(%rax)
1000026fb: c5 fe 7f 40 10              	vmovdqu	%ymm0, 16(%rax)
100002700: 48 8b 44 24 20              	movq	32(%rsp), %rax
100002705: 48 8d 8c 24 e8 00 00 00     	leaq	232(%rsp), %rcx
10000270d: 48 89 8c 24 20 01 00 00     	movq	%rcx, 288(%rsp)
100002715: 48 8d 8c 24 30 01 00 00     	leaq	304(%rsp), %rcx
10000271d: 48 89 8c 24 28 01 00 00     	movq	%rcx, 296(%rsp)
100002725: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002729: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
10000272d: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
100002735: 48 89 df                    	movq	%rbx, %rdi
100002738: be 02 00 00 00              	movl	$2, %esi
10000273d: 4c 89 f2                    	movq	%r14, %rdx
100002740: 31 c9                       	xorl	%ecx, %ecx
100002742: c5 f8 77                    	vzeroupper
100002745: e8 f2 46 00 00              	callq	18162 <dyld_stub_binder+0x100006e3c>
10000274a: 48 8d 9c 24 80 00 00 00     	leaq	128(%rsp), %rbx
100002752: 48 89 df                    	movq	%rbx, %rdi
100002755: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
10000275d: e8 bc 46 00 00              	callq	18108 <dyld_stub_binder+0x100006e1e>
100002762: 48 c7 84 24 70 01 00 00 00 00 00 00 	movq	$0, 368(%rsp)
10000276e: c7 84 24 60 01 00 00 00 00 01 02    	movl	$33619968, 352(%rsp)
100002779: 48 8d 84 24 e0 00 00 00     	leaq	224(%rsp), %rax
100002781: 48 89 84 24 68 01 00 00     	movq	%rax, 360(%rsp)
100002789: 8b 44 24 20                 	movl	32(%rsp), %eax
10000278d: 8b 4c 24 24                 	movl	36(%rsp), %ecx
100002791: 89 8c 24 b0 01 00 00        	movl	%ecx, 432(%rsp)
100002798: 89 84 24 b4 01 00 00        	movl	%eax, 436(%rsp)
10000279f: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000027a3: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
1000027a7: 48 89 df                    	movq	%rbx, %rdi
1000027aa: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
1000027b2: 48 8d 94 24 b0 01 00 00     	leaq	432(%rsp), %rdx
1000027ba: b9 01 00 00 00              	movl	$1, %ecx
1000027bf: e8 90 46 00 00              	callq	18064 <dyld_stub_binder+0x100006e54>
1000027c4: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000027c8: c5 fd 7f 84 24 60 01 00 00  	vmovdqa	%ymm0, 352(%rsp)
1000027d1: c7 84 24 80 00 00 00 00 00 ff 42    	movl	$1124007936, 128(%rsp)
1000027dc: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
1000027e4: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
1000027e9: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
1000027ed: 48 8b 44 24 20              	movq	32(%rsp), %rax
1000027f2: 48 8d 8c 24 88 00 00 00     	leaq	136(%rsp), %rcx
1000027fa: 48 89 8c 24 c0 00 00 00     	movq	%rcx, 192(%rsp)
100002802: 48 8d 8c 24 d0 00 00 00     	leaq	208(%rsp), %rcx
10000280a: 48 89 8c 24 c8 00 00 00     	movq	%rcx, 200(%rsp)
100002812: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002816: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
10000281a: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
100002822: 48 89 df                    	movq	%rbx, %rdi
100002825: be 02 00 00 00              	movl	$2, %esi
10000282a: 4c 89 f2                    	movq	%r14, %rdx
10000282d: b9 10 00 00 00              	movl	$16, %ecx
100002832: c5 f8 77                    	vzeroupper
100002835: e8 02 46 00 00              	callq	17922 <dyld_stub_binder+0x100006e3c>
10000283a: 48 89 df                    	movq	%rbx, %rdi
10000283d: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100002845: e8 fe 45 00 00              	callq	17918 <dyld_stub_binder+0x100006e48>
10000284a: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000284f: 48 85 c0                    	testq	%rax, %rax
100002852: 74 04                       	je	4 <_main+0x3f8>
100002854: f0                          	lock
100002855: ff 40 14                    	incl	20(%rax)
100002858: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100002860: 48 85 c0                    	testq	%rax, %rax
100002863: 74 13                       	je	19 <_main+0x418>
100002865: f0                          	lock
100002866: ff 48 14                    	decl	20(%rax)
100002869: 75 0d                       	jne	13 <_main+0x418>
10000286b: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
100002873: e8 be 45 00 00              	callq	17854 <dyld_stub_binder+0x100006e36>
100002878: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
100002884: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
10000288c: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002890: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100002895: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
10000289d: 0f 8e 2c 06 00 00           	jle	1580 <_main+0xa6f>
1000028a3: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
1000028ab: 31 c9                       	xorl	%ecx, %ecx
1000028ad: 0f 1f 00                    	nopl	(%rax)
1000028b0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000028b7: 48 ff c1                    	incq	%rcx
1000028ba: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
1000028c2: 48 39 d1                    	cmpq	%rdx, %rcx
1000028c5: 7c e9                       	jl	-23 <_main+0x450>
1000028c7: 8b 44 24 18                 	movl	24(%rsp), %eax
1000028cb: 89 84 24 80 00 00 00        	movl	%eax, 128(%rsp)
1000028d2: 83 fa 02                    	cmpl	$2, %edx
1000028d5: 0f 8f 0c 06 00 00           	jg	1548 <_main+0xa87>
1000028db: 8b 44 24 1c                 	movl	28(%rsp), %eax
1000028df: 83 f8 02                    	cmpl	$2, %eax
1000028e2: 0f 8f ff 05 00 00           	jg	1535 <_main+0xa87>
1000028e8: 89 84 24 84 00 00 00        	movl	%eax, 132(%rsp)
1000028ef: 8b 4c 24 20                 	movl	32(%rsp), %ecx
1000028f3: 8b 44 24 24                 	movl	36(%rsp), %eax
1000028f7: 89 8c 24 88 00 00 00        	movl	%ecx, 136(%rsp)
1000028fe: 89 84 24 8c 00 00 00        	movl	%eax, 140(%rsp)
100002905: 48 8b 44 24 60              	movq	96(%rsp), %rax
10000290a: 48 8b 10                    	movq	(%rax), %rdx
10000290d: 48 8b b4 24 c8 00 00 00     	movq	200(%rsp), %rsi
100002915: 48 89 16                    	movq	%rdx, (%rsi)
100002918: 48 8b 40 08                 	movq	8(%rax), %rax
10000291c: 48 89 46 08                 	movq	%rax, 8(%rsi)
100002920: e9 db 05 00 00              	jmp	1499 <_main+0xaa0>
100002925: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000292f: 90                          	nop
100002930: 48 8b 4c 24 58              	movq	88(%rsp), %rcx
100002935: 83 f8 0f                    	cmpl	$15, %eax
100002938: 77 0c                       	ja	12 <_main+0x4e6>
10000293a: be 01 00 00 00              	movl	$1, %esi
10000293f: 31 d2                       	xorl	%edx, %edx
100002941: e9 ea 04 00 00              	jmp	1258 <_main+0x9d0>
100002946: 89 c2                       	movl	%eax, %edx
100002948: 83 e2 f0                    	andl	$-16, %edx
10000294b: 48 8d 72 f0                 	leaq	-16(%rdx), %rsi
10000294f: 48 89 f7                    	movq	%rsi, %rdi
100002952: 48 c1 ef 04                 	shrq	$4, %rdi
100002956: 48 ff c7                    	incq	%rdi
100002959: 89 fb                       	movl	%edi, %ebx
10000295b: 83 e3 03                    	andl	$3, %ebx
10000295e: 48 83 fe 30                 	cmpq	$48, %rsi
100002962: 73 25                       	jae	37 <_main+0x529>
100002964: c4 e2 7d 59 05 1b 47 00 00  	vpbroadcastq	18203(%rip), %ymm0
10000296d: 31 ff                       	xorl	%edi, %edi
10000296f: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
100002973: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100002977: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
10000297b: 48 85 db                    	testq	%rbx, %rbx
10000297e: 0f 85 0e 03 00 00           	jne	782 <_main+0x832>
100002984: e9 d0 03 00 00              	jmp	976 <_main+0x8f9>
100002989: 48 89 de                    	movq	%rbx, %rsi
10000298c: 48 29 fe                    	subq	%rdi, %rsi
10000298f: c4 e2 7d 59 05 f0 46 00 00  	vpbroadcastq	18160(%rip), %ymm0
100002998: 31 ff                       	xorl	%edi, %edi
10000299a: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
10000299e: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
1000029a2: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
1000029a6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000029b0: c4 e2 7d 25 24 b9           	vpmovsxdq	(%rcx,%rdi,4), %ymm4
1000029b6: c4 e2 7d 25 6c b9 10        	vpmovsxdq	16(%rcx,%rdi,4), %ymm5
1000029bd: c4 e2 7d 25 74 b9 20        	vpmovsxdq	32(%rcx,%rdi,4), %ymm6
1000029c4: c4 e2 7d 25 7c b9 30        	vpmovsxdq	48(%rcx,%rdi,4), %ymm7
1000029cb: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
1000029d0: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
1000029d4: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
1000029d9: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
1000029de: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
1000029e3: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000029e9: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000029ed: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000029f2: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
1000029f7: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
1000029fb: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
100002a00: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
100002a05: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
100002a09: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002a0e: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100002a12: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100002a16: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
100002a1b: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
100002a1f: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100002a24: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100002a28: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
100002a2c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002a31: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002a35: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002a39: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
100002a3e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
100002a42: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
100002a47: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
100002a4b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
100002a4f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002a54: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002a58: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100002a5c: c4 e2 7d 25 64 b9 40        	vpmovsxdq	64(%rcx,%rdi,4), %ymm4
100002a63: c4 e2 7d 25 6c b9 50        	vpmovsxdq	80(%rcx,%rdi,4), %ymm5
100002a6a: c4 e2 7d 25 74 b9 60        	vpmovsxdq	96(%rcx,%rdi,4), %ymm6
100002a71: c4 e2 7d 25 7c b9 70        	vpmovsxdq	112(%rcx,%rdi,4), %ymm7
100002a78: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
100002a7d: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100002a82: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100002a87: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
100002a8b: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100002a90: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100002a96: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100002a9a: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100002a9f: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100002aa4: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100002aa8: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
100002aad: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100002ab1: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100002ab6: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002abb: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100002abf: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100002ac3: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100002ac8: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100002acc: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100002ad1: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100002ad5: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002ad9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002ade: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002ae2: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002ae6: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
100002aeb: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100002aef: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100002af4: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100002af8: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002afc: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002b01: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002b05: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100002b09: c4 e2 7d 25 a4 b9 80 00 00 00       	vpmovsxdq	128(%rcx,%rdi,4), %ymm4
100002b13: c4 e2 7d 25 ac b9 90 00 00 00       	vpmovsxdq	144(%rcx,%rdi,4), %ymm5
100002b1d: c4 e2 7d 25 b4 b9 a0 00 00 00       	vpmovsxdq	160(%rcx,%rdi,4), %ymm6
100002b27: c4 e2 7d 25 bc b9 b0 00 00 00       	vpmovsxdq	176(%rcx,%rdi,4), %ymm7
100002b31: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
100002b36: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100002b3b: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100002b40: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
100002b44: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100002b49: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100002b4f: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100002b53: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100002b58: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100002b5d: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100002b61: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
100002b66: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100002b6a: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100002b6f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002b74: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100002b78: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100002b7c: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100002b81: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100002b85: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100002b8a: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100002b8e: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002b92: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002b97: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002b9b: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002b9f: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
100002ba4: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100002ba8: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100002bad: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100002bb1: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002bb5: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002bba: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002bbe: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100002bc2: c4 e2 7d 25 a4 b9 c0 00 00 00       	vpmovsxdq	192(%rcx,%rdi,4), %ymm4
100002bcc: c4 e2 7d 25 ac b9 d0 00 00 00       	vpmovsxdq	208(%rcx,%rdi,4), %ymm5
100002bd6: c4 e2 7d 25 b4 b9 e0 00 00 00       	vpmovsxdq	224(%rcx,%rdi,4), %ymm6
100002be0: c4 e2 7d 25 bc b9 f0 00 00 00       	vpmovsxdq	240(%rcx,%rdi,4), %ymm7
100002bea: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
100002bef: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100002bf4: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100002bf9: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
100002bfd: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100002c02: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100002c08: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100002c0c: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100002c11: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100002c16: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100002c1a: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
100002c1f: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100002c23: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100002c28: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002c2d: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100002c31: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100002c35: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100002c3a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100002c3e: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100002c43: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100002c47: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002c4b: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002c50: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002c54: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002c58: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
100002c5d: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100002c61: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100002c66: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100002c6a: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002c6e: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002c73: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002c77: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100002c7b: 48 83 c7 40                 	addq	$64, %rdi
100002c7f: 48 83 c6 04                 	addq	$4, %rsi
100002c83: 0f 85 27 fd ff ff           	jne	-729 <_main+0x550>
100002c89: 48 85 db                    	testq	%rbx, %rbx
100002c8c: 0f 84 c7 00 00 00           	je	199 <_main+0x8f9>
100002c92: 48 8d 34 b9                 	leaq	(%rcx,%rdi,4), %rsi
100002c96: 48 83 c6 30                 	addq	$48, %rsi
100002c9a: 48 c1 e3 06                 	shlq	$6, %rbx
100002c9e: 31 ff                       	xorl	%edi, %edi
100002ca0: c4 e2 7d 25 64 3e d0        	vpmovsxdq	-48(%rsi,%rdi), %ymm4
100002ca7: c4 e2 7d 25 6c 3e e0        	vpmovsxdq	-32(%rsi,%rdi), %ymm5
100002cae: c4 e2 7d 25 74 3e f0        	vpmovsxdq	-16(%rsi,%rdi), %ymm6
100002cb5: c4 e2 7d 25 3c 3e           	vpmovsxdq	(%rsi,%rdi), %ymm7
100002cbb: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
100002cc0: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
100002cc4: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
100002cc9: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
100002cce: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
100002cd3: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100002cd9: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100002cdd: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100002ce2: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
100002ce7: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
100002ceb: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
100002cf0: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
100002cf5: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
100002cf9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002cfe: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100002d02: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100002d06: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
100002d0b: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
100002d0f: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100002d14: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100002d18: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
100002d1c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002d21: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002d25: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002d29: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
100002d2e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
100002d32: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
100002d37: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
100002d3b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
100002d3f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002d44: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002d48: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100002d4c: 48 83 c7 40                 	addq	$64, %rdi
100002d50: 48 39 fb                    	cmpq	%rdi, %rbx
100002d53: 0f 85 47 ff ff ff           	jne	-185 <_main+0x840>
100002d59: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
100002d5e: c5 dd f4 e0                 	vpmuludq	%ymm0, %ymm4, %ymm4
100002d62: c5 d5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm5
100002d67: c5 e5 f4 ed                 	vpmuludq	%ymm5, %ymm3, %ymm5
100002d6b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
100002d6f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002d74: c5 e5 f4 c0                 	vpmuludq	%ymm0, %ymm3, %ymm0
100002d78: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
100002d7c: c5 e5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm3
100002d81: c5 e5 f4 d8                 	vpmuludq	%ymm0, %ymm3, %ymm3
100002d85: c5 dd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm4
100002d8a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100002d8e: c5 dd d4 db                 	vpaddq	%ymm3, %ymm4, %ymm3
100002d92: c5 e5 73 f3 20              	vpsllq	$32, %ymm3, %ymm3
100002d97: c5 ed f4 c0                 	vpmuludq	%ymm0, %ymm2, %ymm0
100002d9b: c5 fd d4 c3                 	vpaddq	%ymm3, %ymm0, %ymm0
100002d9f: c5 ed 73 d1 20              	vpsrlq	$32, %ymm1, %ymm2
100002da4: c5 ed f4 d0                 	vpmuludq	%ymm0, %ymm2, %ymm2
100002da8: c5 e5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm3
100002dad: c5 f5 f4 db                 	vpmuludq	%ymm3, %ymm1, %ymm3
100002db1: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
100002db5: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
100002dba: c5 f5 f4 c0                 	vpmuludq	%ymm0, %ymm1, %ymm0
100002dbe: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
100002dc2: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100002dc8: c5 ed 73 d0 20              	vpsrlq	$32, %ymm0, %ymm2
100002dcd: c5 ed f4 d1                 	vpmuludq	%ymm1, %ymm2, %ymm2
100002dd1: c5 e5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm3
100002dd6: c5 fd f4 db                 	vpmuludq	%ymm3, %ymm0, %ymm3
100002dda: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
100002dde: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
100002de3: c5 fd f4 c1                 	vpmuludq	%ymm1, %ymm0, %ymm0
100002de7: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
100002deb: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100002df0: c5 e9 73 d0 20              	vpsrlq	$32, %xmm0, %xmm2
100002df5: c5 e9 f4 d1                 	vpmuludq	%xmm1, %xmm2, %xmm2
100002df9: c5 e1 73 d8 0c              	vpsrldq	$12, %xmm0, %xmm3
100002dfe: c5 f9 f4 db                 	vpmuludq	%xmm3, %xmm0, %xmm3
100002e02: c5 e1 d4 d2                 	vpaddq	%xmm2, %xmm3, %xmm2
100002e06: c5 e9 73 f2 20              	vpsllq	$32, %xmm2, %xmm2
100002e0b: c5 f9 f4 c1                 	vpmuludq	%xmm1, %xmm0, %xmm0
100002e0f: c5 f9 d4 c2                 	vpaddq	%xmm2, %xmm0, %xmm0
100002e13: c4 e1 f9 7e c6              	vmovq	%xmm0, %rsi
100002e18: 48 39 c2                    	cmpq	%rax, %rdx
100002e1b: 48 8d 9c 24 10 02 00 00     	leaq	528(%rsp), %rbx
100002e23: 74 1b                       	je	27 <_main+0x9e0>
100002e25: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002e2f: 90                          	nop
100002e30: 48 63 3c 91                 	movslq	(%rcx,%rdx,4), %rdi
100002e34: 48 0f af f7                 	imulq	%rdi, %rsi
100002e38: 48 ff c2                    	incq	%rdx
100002e3b: 48 39 d0                    	cmpq	%rdx, %rax
100002e3e: 75 f0                       	jne	-16 <_main+0x9d0>
100002e40: 85 c0                       	testl	%eax, %eax
100002e42: 0f 85 9c f7 ff ff           	jne	-2148 <_main+0x184>
100002e48: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002e50: 48 8b 44 24 50              	movq	80(%rsp), %rax
100002e55: 48 85 c0                    	testq	%rax, %rax
100002e58: 74 13                       	je	19 <_main+0xa0d>
100002e5a: f0                          	lock
100002e5b: ff 48 14                    	decl	20(%rax)
100002e5e: 75 0d                       	jne	13 <_main+0xa0d>
100002e60: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
100002e65: c5 f8 77                    	vzeroupper
100002e68: e8 c9 3f 00 00              	callq	16329 <dyld_stub_binder+0x100006e36>
100002e6d: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
100002e76: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002e7a: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
100002e7f: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100002e84: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
100002e89: 7e 29                       	jle	41 <_main+0xa54>
100002e8b: 48 8b 44 24 58              	movq	88(%rsp), %rax
100002e90: 31 c9                       	xorl	%ecx, %ecx
100002e92: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002e9c: 0f 1f 40 00                 	nopl	(%rax)
100002ea0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002ea7: 48 ff c1                    	incq	%rcx
100002eaa: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
100002eaf: 48 39 d1                    	cmpq	%rdx, %rcx
100002eb2: 7c ec                       	jl	-20 <_main+0xa40>
100002eb4: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100002eb9: 4c 39 ef                    	cmpq	%r13, %rdi
100002ebc: 0f 84 8e f6 ff ff           	je	-2418 <_main+0xf0>
100002ec2: c5 f8 77                    	vzeroupper
100002ec5: e8 a2 3f 00 00              	callq	16290 <dyld_stub_binder+0x100006e6c>
100002eca: e9 81 f6 ff ff              	jmp	-2431 <_main+0xf0>
100002ecf: 8b 44 24 18                 	movl	24(%rsp), %eax
100002ed3: 89 84 24 80 00 00 00        	movl	%eax, 128(%rsp)
100002eda: 8b 44 24 1c                 	movl	28(%rsp), %eax
100002ede: 83 f8 02                    	cmpl	$2, %eax
100002ee1: 0f 8e 01 fa ff ff           	jle	-1535 <_main+0x488>
100002ee7: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
100002eef: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100002ef4: c5 f8 77                    	vzeroupper
100002ef7: e8 46 3f 00 00              	callq	16198 <dyld_stub_binder+0x100006e42>
100002efc: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100002f00: c4 c1 eb 2a c5              	vcvtsi2sd	%r13, %xmm2, %xmm0
100002f05: c4 c1 eb 2a cf              	vcvtsi2sd	%r15, %xmm2, %xmm1
100002f0a: c5 fb 10 15 6e 41 00 00     	vmovsd	16750(%rip), %xmm2
100002f12: c5 fb 5e c2                 	vdivsd	%xmm2, %xmm0, %xmm0
100002f16: c5 f3 5e ca                 	vdivsd	%xmm2, %xmm1, %xmm1
100002f1a: c5 fc 10 54 24 28           	vmovups	40(%rsp), %ymm2
100002f20: c5 fc 11 94 24 90 00 00 00  	vmovups	%ymm2, 144(%rsp)
100002f29: c5 f9 10 54 24 48           	vmovupd	72(%rsp), %xmm2
100002f2f: c5 f9 11 94 24 b0 00 00 00  	vmovupd	%xmm2, 176(%rsp)
100002f38: 85 c9                       	testl	%ecx, %ecx
100002f3a: 4d 89 f5                    	movq	%r14, %r13
100002f3d: 0f 84 53 01 00 00           	je	339 <_main+0xc36>
100002f43: 31 c0                       	xorl	%eax, %eax
100002f45: 8b 74 24 24                 	movl	36(%rsp), %esi
100002f49: 85 f6                       	testl	%esi, %esi
100002f4b: be 00 00 00 00              	movl	$0, %esi
100002f50: 75 21                       	jne	33 <_main+0xb13>
100002f52: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002f5c: 0f 1f 40 00                 	nopl	(%rax)
100002f60: ff c0                       	incl	%eax
100002f62: 39 c8                       	cmpl	%ecx, %eax
100002f64: 0f 83 2c 01 00 00           	jae	300 <_main+0xc36>
100002f6a: 85 f6                       	testl	%esi, %esi
100002f6c: be 00 00 00 00              	movl	$0, %esi
100002f71: 74 ed                       	je	-19 <_main+0xb00>
100002f73: 48 63 c8                    	movslq	%eax, %rcx
100002f76: 31 d2                       	xorl	%edx, %edx
100002f78: c5 fb 10 25 18 41 00 00     	vmovsd	16664(%rip), %xmm4
100002f80: c5 fa 10 2d 48 41 00 00     	vmovss	16712(%rip), %xmm5
100002f88: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002f90: 48 8b 74 24 60              	movq	96(%rsp), %rsi
100002f95: 48 8b 3e                    	movq	(%rsi), %rdi
100002f98: 48 0f af f9                 	imulq	%rcx, %rdi
100002f9c: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100002fa1: 48 63 d2                    	movslq	%edx, %rdx
100002fa4: 48 8d 34 52                 	leaq	(%rdx,%rdx,2), %rsi
100002fa8: 0f b6 3c 37                 	movzbl	(%rdi,%rsi), %edi
100002fac: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100002fb0: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100002fb4: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100002fb8: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100002fc0: 48 8b 1b                    	movq	(%rbx), %rbx
100002fc3: 48 0f af d9                 	imulq	%rcx, %rbx
100002fc7: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100002fcf: 40 88 3c 33                 	movb	%dil, (%rbx,%rsi)
100002fd3: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100002fd8: 48 8b 3f                    	movq	(%rdi), %rdi
100002fdb: 48 0f af f9                 	imulq	%rcx, %rdi
100002fdf: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100002fe4: 0f b6 7c 37 01              	movzbl	1(%rdi,%rsi), %edi
100002fe9: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100002fed: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
100002ff5: 48 8b 3f                    	movq	(%rdi), %rdi
100002ff8: 48 0f af f9                 	imulq	%rcx, %rdi
100002ffc: 48 03 bc 24 f0 00 00 00     	addq	240(%rsp), %rdi
100003004: 0f b6 3c 3a                 	movzbl	(%rdx,%rdi), %edi
100003008: c5 ca 2a df                 	vcvtsi2ss	%edi, %xmm6, %xmm3
10000300c: c5 e2 59 dd                 	vmulss	%xmm5, %xmm3, %xmm3
100003010: c5 e2 5a db                 	vcvtss2sd	%xmm3, %xmm3, %xmm3
100003014: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003018: c5 eb 58 d3                 	vaddsd	%xmm3, %xmm2, %xmm2
10000301c: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003020: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100003028: 48 8b 1b                    	movq	(%rbx), %rbx
10000302b: 48 0f af d9                 	imulq	%rcx, %rbx
10000302f: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100003037: 40 88 7c 33 01              	movb	%dil, 1(%rbx,%rsi)
10000303c: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003041: 48 8b 3f                    	movq	(%rdi), %rdi
100003044: 48 0f af f9                 	imulq	%rcx, %rdi
100003048: 48 03 7c 24 28              	addq	40(%rsp), %rdi
10000304d: 0f b6 7c 37 02              	movzbl	2(%rdi,%rsi), %edi
100003052: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003056: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
10000305a: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
10000305e: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100003066: 48 8b 1b                    	movq	(%rbx), %rbx
100003069: 48 0f af d9                 	imulq	%rcx, %rbx
10000306d: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100003075: 40 88 7c 33 02              	movb	%dil, 2(%rbx,%rsi)
10000307a: ff c2                       	incl	%edx
10000307c: 8b 74 24 24                 	movl	36(%rsp), %esi
100003080: 39 f2                       	cmpl	%esi, %edx
100003082: 0f 82 08 ff ff ff           	jb	-248 <_main+0xb30>
100003088: 8b 4c 24 20                 	movl	32(%rsp), %ecx
10000308c: ff c0                       	incl	%eax
10000308e: 39 c8                       	cmpl	%ecx, %eax
100003090: 0f 82 d4 fe ff ff           	jb	-300 <_main+0xb0a>
100003096: c5 fb 10 15 02 40 00 00     	vmovsd	16386(%rip), %xmm2
10000309e: c5 eb 59 54 24 78           	vmulsd	120(%rsp), %xmm2, %xmm2
1000030a4: c5 f3 5c c0                 	vsubsd	%xmm0, %xmm1, %xmm0
1000030a8: c5 fb 58 05 f8 3f 00 00     	vaddsd	16376(%rip), %xmm0, %xmm0
1000030b0: c5 fb 10 0d f8 3f 00 00     	vmovsd	16376(%rip), %xmm1
1000030b8: c5 f3 5e c0                 	vdivsd	%xmm0, %xmm1, %xmm0
1000030bc: c5 eb 58 c0                 	vaddsd	%xmm0, %xmm2, %xmm0
1000030c0: 8b 9c 24 f8 01 00 00        	movl	504(%rsp), %ebx
1000030c7: c5 fb 11 44 24 78           	vmovsd	%xmm0, 120(%rsp)
1000030cd: c5 f8 77                    	vzeroupper
1000030d0: e8 27 3e 00 00              	callq	15911 <dyld_stub_binder+0x100006efc>
1000030d5: c5 fb 2c f0                 	vcvttsd2si	%xmm0, %esi
1000030d9: 4c 89 e7                    	movq	%r12, %rdi
1000030dc: e8 e5 3d 00 00              	callq	15845 <dyld_stub_binder+0x100006ec6>
1000030e1: 4c 89 e7                    	movq	%r12, %rdi
1000030e4: 31 f6                       	xorl	%esi, %esi
1000030e6: 48 8d 15 18 5e 00 00        	leaq	24088(%rip), %rdx
1000030ed: e8 a4 3d 00 00              	callq	15780 <dyld_stub_binder+0x100006e96>
1000030f2: 48 8b 48 10                 	movq	16(%rax), %rcx
1000030f6: 48 89 8c 24 50 01 00 00     	movq	%rcx, 336(%rsp)
1000030fe: c5 f9 10 00                 	vmovupd	(%rax), %xmm0
100003102: c5 f9 29 84 24 40 01 00 00  	vmovapd	%xmm0, 320(%rsp)
10000310b: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
10000310f: c5 f9 11 00                 	vmovupd	%xmm0, (%rax)
100003113: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
10000311b: 48 8d bc 24 40 01 00 00     	leaq	320(%rsp), %rdi
100003123: 48 8d 35 e2 5d 00 00        	leaq	24034(%rip), %rsi
10000312a: e8 5b 3d 00 00              	callq	15707 <dyld_stub_binder+0x100006e8a>
10000312f: c4 e1 cb 2a c3              	vcvtsi2sd	%rbx, %xmm6, %xmm0
100003134: c5 fb 59 44 24 78           	vmulsd	120(%rsp), %xmm0, %xmm0
10000313a: c5 fb 5e 05 76 3f 00 00     	vdivsd	16246(%rip), %xmm0, %xmm0
100003142: 48 8b 48 10                 	movq	16(%rax), %rcx
100003146: 48 89 8c 24 d0 03 00 00     	movq	%rcx, 976(%rsp)
10000314e: c5 f9 10 08                 	vmovupd	(%rax), %xmm1
100003152: c5 f9 29 8c 24 c0 03 00 00  	vmovapd	%xmm1, 960(%rsp)
10000315b: c5 f1 57 c9                 	vxorpd	%xmm1, %xmm1, %xmm1
10000315f: c5 f9 11 08                 	vmovupd	%xmm1, (%rax)
100003163: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
10000316b: 48 8d bc 24 98 01 00 00     	leaq	408(%rsp), %rdi
100003173: e8 48 3d 00 00              	callq	15688 <dyld_stub_binder+0x100006ec0>
100003178: 0f b6 94 24 98 01 00 00     	movzbl	408(%rsp), %edx
100003180: f6 c2 01                    	testb	$1, %dl
100003183: 48 8d 9c 24 10 02 00 00     	leaq	528(%rsp), %rbx
10000318b: 74 12                       	je	18 <_main+0xd3f>
10000318d: 48 8b b4 24 a8 01 00 00     	movq	424(%rsp), %rsi
100003195: 48 8b 94 24 a0 01 00 00     	movq	416(%rsp), %rdx
10000319d: eb 0b                       	jmp	11 <_main+0xd4a>
10000319f: 48 d1 ea                    	shrq	%rdx
1000031a2: 48 8d b4 24 99 01 00 00     	leaq	409(%rsp), %rsi
1000031aa: 4c 89 ef                    	movq	%r13, %rdi
1000031ad: e8 de 3c 00 00              	callq	15582 <dyld_stub_binder+0x100006e90>
1000031b2: 48 8b 48 10                 	movq	16(%rax), %rcx
1000031b6: 48 89 8c 24 70 01 00 00     	movq	%rcx, 368(%rsp)
1000031be: c5 f8 10 00                 	vmovups	(%rax), %xmm0
1000031c2: c5 f8 29 84 24 60 01 00 00  	vmovaps	%xmm0, 352(%rsp)
1000031cb: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000031cf: c5 f8 11 00                 	vmovups	%xmm0, (%rax)
1000031d3: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
1000031db: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
1000031e3: 0f 85 80 01 00 00           	jne	384 <_main+0xf09>
1000031e9: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000031f1: 0f 85 8d 01 00 00           	jne	397 <_main+0xf24>
1000031f7: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
1000031ff: 0f 85 9a 01 00 00           	jne	410 <_main+0xf3f>
100003205: 4d 89 e7                    	movq	%r12, %r15
100003208: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003210: 74 0d                       	je	13 <_main+0xdbf>
100003212: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
10000321a: e8 b3 3c 00 00              	callq	15539 <dyld_stub_binder+0x100006ed2>
10000321f: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
10000322b: c7 84 24 c0 03 00 00 00 00 01 03    	movl	$50397184, 960(%rsp)
100003236: 4c 8d a4 24 80 00 00 00     	leaq	128(%rsp), %r12
10000323e: 4c 89 a4 24 c8 03 00 00     	movq	%r12, 968(%rsp)
100003246: 48 b8 1e 00 00 00 1e 00 00 00       	movabsq	$128849018910, %rax
100003250: 48 89 84 24 b8 01 00 00     	movq	%rax, 440(%rsp)
100003258: c5 fc 28 05 a0 3e 00 00     	vmovaps	16032(%rip), %ymm0
100003260: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100003269: c7 44 24 08 00 00 00 00     	movl	$0, 8(%rsp)
100003271: c7 04 24 10 00 00 00        	movl	$16, (%rsp)
100003278: 4c 89 ef                    	movq	%r13, %rdi
10000327b: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003283: 48 8d 94 24 b8 01 00 00     	leaq	440(%rsp), %rdx
10000328b: 31 c9                       	xorl	%ecx, %ecx
10000328d: c5 fb 10 05 2b 3e 00 00     	vmovsd	15915(%rip), %xmm0
100003295: 4c 8d 84 24 40 02 00 00     	leaq	576(%rsp), %r8
10000329d: 41 b9 02 00 00 00           	movl	$2, %r9d
1000032a3: c5 f8 77                    	vzeroupper
1000032a6: e8 af 3b 00 00              	callq	15279 <dyld_stub_binder+0x100006e5a>
1000032ab: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
1000032af: c5 f9 29 84 24 c0 03 00 00  	vmovapd	%xmm0, 960(%rsp)
1000032b8: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
1000032c4: c6 84 24 c0 03 00 00 0a     	movb	$10, 960(%rsp)
1000032cc: 48 8d 84 24 c1 03 00 00     	leaq	961(%rsp), %rax
1000032d4: c6 40 04 65                 	movb	$101, 4(%rax)
1000032d8: c7 00 66 72 61 6d           	movl	$1835102822, (%rax)
1000032de: c6 84 24 c6 03 00 00 00     	movb	$0, 966(%rsp)
1000032e6: 48 c7 84 24 50 01 00 00 00 00 00 00 	movq	$0, 336(%rsp)
1000032f2: c7 84 24 40 01 00 00 00 00 01 01    	movl	$16842752, 320(%rsp)
1000032fd: 4c 89 a4 24 48 01 00 00     	movq	%r12, 328(%rsp)
100003305: 4c 89 ef                    	movq	%r13, %rdi
100003308: 48 8d b4 24 40 01 00 00     	leaq	320(%rsp), %rsi
100003310: e8 39 3b 00 00              	callq	15161 <dyld_stub_binder+0x100006e4e>
100003315: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
10000331d: 4d 89 fc                    	movq	%r15, %r12
100003320: 4c 8d 6c 24 68              	leaq	104(%rsp), %r13
100003325: 0f 85 97 00 00 00           	jne	151 <_main+0xf62>
10000332b: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003333: 4c 8d bc 24 80 00 00 00     	leaq	128(%rsp), %r15
10000333b: 0f 85 a4 00 00 00           	jne	164 <_main+0xf85>
100003341: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003349: 48 85 c0                    	testq	%rax, %rax
10000334c: 0f 84 b1 00 00 00           	je	177 <_main+0xfa3>
100003352: f0                          	lock
100003353: ff 48 14                    	decl	20(%rax)
100003356: 0f 85 a7 00 00 00           	jne	167 <_main+0xfa3>
10000335c: 4c 89 ff                    	movq	%r15, %rdi
10000335f: e8 d2 3a 00 00              	callq	15058 <dyld_stub_binder+0x100006e36>
100003364: e9 9a 00 00 00              	jmp	154 <_main+0xfa3>
100003369: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100003371: e8 5c 3b 00 00              	callq	15196 <dyld_stub_binder+0x100006ed2>
100003376: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
10000337e: 0f 84 73 fe ff ff           	je	-397 <_main+0xd97>
100003384: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
10000338c: e8 41 3b 00 00              	callq	15169 <dyld_stub_binder+0x100006ed2>
100003391: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003399: 0f 84 66 fe ff ff           	je	-410 <_main+0xda5>
10000339f: 48 8b bc 24 50 01 00 00     	movq	336(%rsp), %rdi
1000033a7: e8 26 3b 00 00              	callq	15142 <dyld_stub_binder+0x100006ed2>
1000033ac: 4d 89 e7                    	movq	%r12, %r15
1000033af: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
1000033b7: 0f 85 55 fe ff ff           	jne	-427 <_main+0xdb2>
1000033bd: e9 5d fe ff ff              	jmp	-419 <_main+0xdbf>
1000033c2: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
1000033ca: e8 03 3b 00 00              	callq	15107 <dyld_stub_binder+0x100006ed2>
1000033cf: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
1000033d7: 4c 8d bc 24 80 00 00 00     	leaq	128(%rsp), %r15
1000033df: 0f 84 5c ff ff ff           	je	-164 <_main+0xee1>
1000033e5: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
1000033ed: e8 e0 3a 00 00              	callq	15072 <dyld_stub_binder+0x100006ed2>
1000033f2: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000033fa: 48 85 c0                    	testq	%rax, %rax
1000033fd: 0f 85 4f ff ff ff           	jne	-177 <_main+0xef2>
100003403: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
10000340f: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
100003417: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
10000341b: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100003420: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
100003428: 7e 2d                       	jle	45 <_main+0xff7>
10000342a: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
100003432: 31 c9                       	xorl	%ecx, %ecx
100003434: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000343e: 66 90                       	nop
100003440: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003447: 48 ff c1                    	incq	%rcx
10000344a: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
100003452: 48 39 d1                    	cmpq	%rdx, %rcx
100003455: 7c e9                       	jl	-23 <_main+0xfe0>
100003457: 48 8b bc 24 c8 00 00 00     	movq	200(%rsp), %rdi
10000345f: 48 8d 84 24 d0 00 00 00     	leaq	208(%rsp), %rax
100003467: 48 39 c7                    	cmpq	%rax, %rdi
10000346a: 74 08                       	je	8 <_main+0x1014>
10000346c: c5 f8 77                    	vzeroupper
10000346f: e8 f8 39 00 00              	callq	14840 <dyld_stub_binder+0x100006e6c>
100003474: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
10000347c: 48 85 c0                    	testq	%rax, %rax
10000347f: 74 16                       	je	22 <_main+0x1037>
100003481: f0                          	lock
100003482: ff 48 14                    	decl	20(%rax)
100003485: 75 10                       	jne	16 <_main+0x1037>
100003487: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
10000348f: c5 f8 77                    	vzeroupper
100003492: e8 9f 39 00 00              	callq	14751 <dyld_stub_binder+0x100006e36>
100003497: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
1000034a3: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
1000034ab: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
1000034af: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
1000034b3: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
1000034bb: 7e 2a                       	jle	42 <_main+0x1087>
1000034bd: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
1000034c5: 31 c9                       	xorl	%ecx, %ecx
1000034c7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
1000034d0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000034d7: 48 ff c1                    	incq	%rcx
1000034da: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
1000034e2: 48 39 d1                    	cmpq	%rdx, %rcx
1000034e5: 7c e9                       	jl	-23 <_main+0x1070>
1000034e7: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
1000034ef: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
1000034f7: 48 39 c7                    	cmpq	%rax, %rdi
1000034fa: 74 08                       	je	8 <_main+0x10a4>
1000034fc: c5 f8 77                    	vzeroupper
1000034ff: e8 68 39 00 00              	callq	14696 <dyld_stub_binder+0x100006e6c>
100003504: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
10000350c: c5 f8 77                    	vzeroupper
10000350f: e8 0c 05 00 00              	callq	1292 <_main+0x15c0>
100003514: 45 31 ff                    	xorl	%r15d, %r15d
100003517: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000351c: 48 85 c0                    	testq	%rax, %rax
10000351f: 0f 85 35 f9 ff ff           	jne	-1739 <_main+0x9fa>
100003525: e9 43 f9 ff ff              	jmp	-1725 <_main+0xa0d>
10000352a: 48 8b 3d 0f 5b 00 00        	movq	23311(%rip), %rdi
100003531: 48 8d 35 e8 59 00 00        	leaq	23016(%rip), %rsi
100003538: ba 0d 00 00 00              	movl	$13, %edx
10000353d: c5 f8 77                    	vzeroupper
100003540: e8 0b 07 00 00              	callq	1803 <_main+0x17f0>
100003545: 48 8d bc 24 10 02 00 00     	leaq	528(%rsp), %rdi
10000354d: e8 d8 38 00 00              	callq	14552 <dyld_stub_binder+0x100006e2a>
100003552: 48 8b 05 f7 5a 00 00        	movq	23287(%rip), %rax
100003559: 48 83 c0 10                 	addq	$16, %rax
10000355d: 48 89 84 24 d8 01 00 00     	movq	%rax, 472(%rsp)
100003565: 48 8b bc 24 00 02 00 00     	movq	512(%rsp), %rdi
10000356d: 48 85 ff                    	testq	%rdi, %rdi
100003570: 74 05                       	je	5 <_main+0x1117>
100003572: e8 5b 39 00 00              	callq	14683 <dyld_stub_binder+0x100006ed2>
100003577: 48 8b bc 24 08 02 00 00     	movq	520(%rsp), %rdi
10000357f: 48 85 ff                    	testq	%rdi, %rdi
100003582: 74 05                       	je	5 <_main+0x1129>
100003584: e8 49 39 00 00              	callq	14665 <dyld_stub_binder+0x100006ed2>
100003589: 48 8b 05 d0 5a 00 00        	movq	23248(%rip), %rax
100003590: 48 8b 00                    	movq	(%rax), %rax
100003593: 48 3b 84 24 e0 03 00 00     	cmpq	992(%rsp), %rax
10000359b: 75 11                       	jne	17 <_main+0x114e>
10000359d: 31 c0                       	xorl	%eax, %eax
10000359f: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
1000035a3: 5b                          	popq	%rbx
1000035a4: 41 5c                       	popq	%r12
1000035a6: 41 5d                       	popq	%r13
1000035a8: 41 5e                       	popq	%r14
1000035aa: 41 5f                       	popq	%r15
1000035ac: 5d                          	popq	%rbp
1000035ad: c3                          	retq
1000035ae: e8 3d 39 00 00              	callq	14653 <dyld_stub_binder+0x100006ef0>
1000035b3: e9 ed 03 00 00              	jmp	1005 <_main+0x1545>
1000035b8: 48 89 c3                    	movq	%rax, %rbx
1000035bb: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
1000035c3: 0f 84 ef 03 00 00           	je	1007 <_main+0x1558>
1000035c9: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
1000035d1: e8 fc 38 00 00              	callq	14588 <dyld_stub_binder+0x100006ed2>
1000035d6: e9 dd 03 00 00              	jmp	989 <_main+0x1558>
1000035db: 48 89 c3                    	movq	%rax, %rbx
1000035de: e9 d5 03 00 00              	jmp	981 <_main+0x1558>
1000035e3: 48 89 c7                    	movq	%rax, %rdi
1000035e6: e8 25 04 00 00              	callq	1061 <_main+0x15b0>
1000035eb: 48 89 c7                    	movq	%rax, %rdi
1000035ee: e8 1d 04 00 00              	callq	1053 <_main+0x15b0>
1000035f3: 48 89 c7                    	movq	%rax, %rdi
1000035f6: e8 15 04 00 00              	callq	1045 <_main+0x15b0>
1000035fb: 48 89 c3                    	movq	%rax, %rbx
1000035fe: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003606: 48 85 c0                    	testq	%rax, %rax
100003609: 0f 85 c8 01 00 00           	jne	456 <_main+0x1377>
10000360f: e9 d6 01 00 00              	jmp	470 <_main+0x138a>
100003614: 48 89 c3                    	movq	%rax, %rbx
100003617: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
10000361f: 48 85 c0                    	testq	%rax, %rax
100003622: 74 13                       	je	19 <_main+0x11d7>
100003624: f0                          	lock
100003625: ff 48 14                    	decl	20(%rax)
100003628: 75 0d                       	jne	13 <_main+0x11d7>
10000362a: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003632: e8 ff 37 00 00              	callq	14335 <dyld_stub_binder+0x100006e36>
100003637: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100003643: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003647: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
10000364f: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100003653: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
10000365b: 7e 21                       	jle	33 <_main+0x121e>
10000365d: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100003665: 31 c9                       	xorl	%ecx, %ecx
100003667: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
10000366e: 48 ff c1                    	incq	%rcx
100003671: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
100003679: 48 39 d1                    	cmpq	%rdx, %rcx
10000367c: 7c e9                       	jl	-23 <_main+0x1207>
10000367e: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
100003686: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
10000368e: 48 39 c7                    	cmpq	%rax, %rdi
100003691: 0f 84 96 02 00 00           	je	662 <_main+0x14cd>
100003697: c5 f8 77                    	vzeroupper
10000369a: e8 cd 37 00 00              	callq	14285 <dyld_stub_binder+0x100006e6c>
10000369f: e9 89 02 00 00              	jmp	649 <_main+0x14cd>
1000036a4: 48 89 c7                    	movq	%rax, %rdi
1000036a7: e8 64 03 00 00              	callq	868 <_main+0x15b0>
1000036ac: 48 89 c3                    	movq	%rax, %rbx
1000036af: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000036b4: 48 85 c0                    	testq	%rax, %rax
1000036b7: 0f 85 7a 02 00 00           	jne	634 <_main+0x14d7>
1000036bd: e9 88 02 00 00              	jmp	648 <_main+0x14ea>
1000036c2: 48 89 c3                    	movq	%rax, %rbx
1000036c5: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000036cd: 74 1f                       	je	31 <_main+0x128e>
1000036cf: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
1000036d7: e8 f6 37 00 00              	callq	14326 <dyld_stub_binder+0x100006ed2>
1000036dc: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
1000036e4: 75 16                       	jne	22 <_main+0x129c>
1000036e6: e9 df 00 00 00              	jmp	223 <_main+0x136a>
1000036eb: 48 89 c3                    	movq	%rax, %rbx
1000036ee: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
1000036f6: 0f 84 ce 00 00 00           	je	206 <_main+0x136a>
1000036fc: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
100003704: e9 aa 00 00 00              	jmp	170 <_main+0x1353>
100003709: 48 89 c3                    	movq	%rax, %rbx
10000370c: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
100003714: 75 23                       	jne	35 <_main+0x12d9>
100003716: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
10000371e: 75 3f                       	jne	63 <_main+0x12ff>
100003720: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003728: 75 5b                       	jne	91 <_main+0x1325>
10000372a: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003732: 75 77                       	jne	119 <_main+0x134b>
100003734: e9 91 00 00 00              	jmp	145 <_main+0x136a>
100003739: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100003741: e8 8c 37 00 00              	callq	14220 <dyld_stub_binder+0x100006ed2>
100003746: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
10000374e: 74 d0                       	je	-48 <_main+0x12c0>
100003750: eb 0d                       	jmp	13 <_main+0x12ff>
100003752: 48 89 c3                    	movq	%rax, %rbx
100003755: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
10000375d: 74 c1                       	je	-63 <_main+0x12c0>
10000375f: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003767: e8 66 37 00 00              	callq	14182 <dyld_stub_binder+0x100006ed2>
10000376c: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003774: 74 b4                       	je	-76 <_main+0x12ca>
100003776: eb 0d                       	jmp	13 <_main+0x1325>
100003778: 48 89 c3                    	movq	%rax, %rbx
10000377b: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003783: 74 a5                       	je	-91 <_main+0x12ca>
100003785: 48 8b bc 24 50 01 00 00     	movq	336(%rsp), %rdi
10000378d: e8 40 37 00 00              	callq	14144 <dyld_stub_binder+0x100006ed2>
100003792: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
10000379a: 75 0f                       	jne	15 <_main+0x134b>
10000379c: eb 2c                       	jmp	44 <_main+0x136a>
10000379e: 48 89 c3                    	movq	%rax, %rbx
1000037a1: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
1000037a9: 74 1f                       	je	31 <_main+0x136a>
1000037ab: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
1000037b3: e8 1a 37 00 00              	callq	14106 <dyld_stub_binder+0x100006ed2>
1000037b8: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000037c0: 48 85 c0                    	testq	%rax, %rax
1000037c3: 75 12                       	jne	18 <_main+0x1377>
1000037c5: eb 23                       	jmp	35 <_main+0x138a>
1000037c7: 48 89 c3                    	movq	%rax, %rbx
1000037ca: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000037d2: 48 85 c0                    	testq	%rax, %rax
1000037d5: 74 13                       	je	19 <_main+0x138a>
1000037d7: f0                          	lock
1000037d8: ff 48 14                    	decl	20(%rax)
1000037db: 75 0d                       	jne	13 <_main+0x138a>
1000037dd: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
1000037e5: e8 4c 36 00 00              	callq	13900 <dyld_stub_binder+0x100006e36>
1000037ea: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
1000037f6: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
1000037fa: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
100003802: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100003807: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
10000380f: 7e 21                       	jle	33 <_main+0x13d2>
100003811: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
100003819: 31 c9                       	xorl	%ecx, %ecx
10000381b: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003822: 48 ff c1                    	incq	%rcx
100003825: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
10000382d: 48 39 d1                    	cmpq	%rdx, %rcx
100003830: 7c e9                       	jl	-23 <_main+0x13bb>
100003832: 48 8b bc 24 c8 00 00 00     	movq	200(%rsp), %rdi
10000383a: 48 8d 84 24 d0 00 00 00     	leaq	208(%rsp), %rax
100003842: 48 39 c7                    	cmpq	%rax, %rdi
100003845: 74 21                       	je	33 <_main+0x1408>
100003847: c5 f8 77                    	vzeroupper
10000384a: e8 1d 36 00 00              	callq	13853 <dyld_stub_binder+0x100006e6c>
10000384f: eb 17                       	jmp	23 <_main+0x1408>
100003851: 48 89 c7                    	movq	%rax, %rdi
100003854: e8 b7 01 00 00              	callq	439 <_main+0x15b0>
100003859: eb 0a                       	jmp	10 <_main+0x1405>
10000385b: eb 08                       	jmp	8 <_main+0x1405>
10000385d: 48 89 c3                    	movq	%rax, %rbx
100003860: e9 8a 00 00 00              	jmp	138 <_main+0x148f>
100003865: 48 89 c3                    	movq	%rax, %rbx
100003868: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
100003870: 48 85 c0                    	testq	%rax, %rax
100003873: 74 16                       	je	22 <_main+0x142b>
100003875: f0                          	lock
100003876: ff 48 14                    	decl	20(%rax)
100003879: 75 10                       	jne	16 <_main+0x142b>
10000387b: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003883: c5 f8 77                    	vzeroupper
100003886: e8 ab 35 00 00              	callq	13739 <dyld_stub_binder+0x100006e36>
10000388b: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100003897: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
10000389b: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
1000038a3: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
1000038a7: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
1000038af: 7e 21                       	jle	33 <_main+0x1472>
1000038b1: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
1000038b9: 31 c9                       	xorl	%ecx, %ecx
1000038bb: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000038c2: 48 ff c1                    	incq	%rcx
1000038c5: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
1000038cd: 48 39 d1                    	cmpq	%rdx, %rcx
1000038d0: 7c e9                       	jl	-23 <_main+0x145b>
1000038d2: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
1000038da: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
1000038e2: 48 39 c7                    	cmpq	%rax, %rdi
1000038e5: 74 08                       	je	8 <_main+0x148f>
1000038e7: c5 f8 77                    	vzeroupper
1000038ea: e8 7d 35 00 00              	callq	13693 <dyld_stub_binder+0x100006e6c>
1000038ef: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
1000038f7: c5 f8 77                    	vzeroupper
1000038fa: e8 21 01 00 00              	callq	289 <_main+0x15c0>
1000038ff: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003904: 48 85 c0                    	testq	%rax, %rax
100003907: 75 2e                       	jne	46 <_main+0x14d7>
100003909: eb 3f                       	jmp	63 <_main+0x14ea>
10000390b: 48 89 c7                    	movq	%rax, %rdi
10000390e: e8 fd 00 00 00              	callq	253 <_main+0x15b0>
100003913: 48 89 c3                    	movq	%rax, %rbx
100003916: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000391b: 48 85 c0                    	testq	%rax, %rax
10000391e: 75 17                       	jne	23 <_main+0x14d7>
100003920: eb 28                       	jmp	40 <_main+0x14ea>
100003922: 48 89 c7                    	movq	%rax, %rdi
100003925: e8 e6 00 00 00              	callq	230 <_main+0x15b0>
10000392a: 48 89 c3                    	movq	%rax, %rbx
10000392d: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003932: 48 85 c0                    	testq	%rax, %rax
100003935: 74 13                       	je	19 <_main+0x14ea>
100003937: f0                          	lock
100003938: ff 48 14                    	decl	20(%rax)
10000393b: 75 0d                       	jne	13 <_main+0x14ea>
10000393d: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
100003942: c5 f8 77                    	vzeroupper
100003945: e8 ec 34 00 00              	callq	13548 <dyld_stub_binder+0x100006e36>
10000394a: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
100003953: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003957: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
10000395c: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100003961: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
100003966: 7e 1c                       	jle	28 <_main+0x1524>
100003968: 48 8b 44 24 58              	movq	88(%rsp), %rax
10000396d: 31 c9                       	xorl	%ecx, %ecx
10000396f: 90                          	nop
100003970: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003977: 48 ff c1                    	incq	%rcx
10000397a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000397f: 48 39 d1                    	cmpq	%rdx, %rcx
100003982: 7c ec                       	jl	-20 <_main+0x1510>
100003984: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003989: 48 8d 44 24 68              	leaq	104(%rsp), %rax
10000398e: 48 39 c7                    	cmpq	%rax, %rdi
100003991: 74 15                       	je	21 <_main+0x1548>
100003993: c5 f8 77                    	vzeroupper
100003996: e8 d1 34 00 00              	callq	13521 <dyld_stub_binder+0x100006e6c>
10000399b: eb 0b                       	jmp	11 <_main+0x1548>
10000399d: 48 89 c7                    	movq	%rax, %rdi
1000039a0: e8 6b 00 00 00              	callq	107 <_main+0x15b0>
1000039a5: 48 89 c3                    	movq	%rax, %rbx
1000039a8: 48 8d bc 24 10 02 00 00     	leaq	528(%rsp), %rdi
1000039b0: c5 f8 77                    	vzeroupper
1000039b3: e8 72 34 00 00              	callq	13426 <dyld_stub_binder+0x100006e2a>
1000039b8: 48 8b 05 91 56 00 00        	movq	22161(%rip), %rax
1000039bf: 48 83 c0 10                 	addq	$16, %rax
1000039c3: 48 89 84 24 d8 01 00 00     	movq	%rax, 472(%rsp)
1000039cb: 48 8b bc 24 00 02 00 00     	movq	512(%rsp), %rdi
1000039d3: 48 85 ff                    	testq	%rdi, %rdi
1000039d6: 75 17                       	jne	23 <_main+0x158f>
1000039d8: 48 8b bc 24 08 02 00 00     	movq	520(%rsp), %rdi
1000039e0: 48 85 ff                    	testq	%rdi, %rdi
1000039e3: 75 1c                       	jne	28 <_main+0x15a1>
1000039e5: 48 89 df                    	movq	%rbx, %rdi
1000039e8: e8 2b 34 00 00              	callq	13355 <dyld_stub_binder+0x100006e18>
1000039ed: 0f 0b                       	ud2
1000039ef: e8 de 34 00 00              	callq	13534 <dyld_stub_binder+0x100006ed2>
1000039f4: 48 8b bc 24 08 02 00 00     	movq	520(%rsp), %rdi
1000039fc: 48 85 ff                    	testq	%rdi, %rdi
1000039ff: 74 e4                       	je	-28 <_main+0x1585>
100003a01: e8 cc 34 00 00              	callq	13516 <dyld_stub_binder+0x100006ed2>
100003a06: 48 89 df                    	movq	%rbx, %rdi
100003a09: e8 0a 34 00 00              	callq	13322 <dyld_stub_binder+0x100006e18>
100003a0e: 0f 0b                       	ud2
100003a10: 50                          	pushq	%rax
100003a11: e8 ce 34 00 00              	callq	13518 <dyld_stub_binder+0x100006ee4>
100003a16: e8 b1 34 00 00              	callq	13489 <dyld_stub_binder+0x100006ecc>
100003a1b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003a20: 55                          	pushq	%rbp
100003a21: 48 89 e5                    	movq	%rsp, %rbp
100003a24: 53                          	pushq	%rbx
100003a25: 50                          	pushq	%rax
100003a26: 48 89 fb                    	movq	%rdi, %rbx
100003a29: 48 8b 87 08 01 00 00        	movq	264(%rdi), %rax
100003a30: 48 85 c0                    	testq	%rax, %rax
100003a33: 74 12                       	je	18 <_main+0x15e7>
100003a35: f0                          	lock
100003a36: ff 48 14                    	decl	20(%rax)
100003a39: 75 0c                       	jne	12 <_main+0x15e7>
100003a3b: 48 8d bb d0 00 00 00        	leaq	208(%rbx), %rdi
100003a42: e8 ef 33 00 00              	callq	13295 <dyld_stub_binder+0x100006e36>
100003a47: 48 c7 83 08 01 00 00 00 00 00 00    	movq	$0, 264(%rbx)
100003a52: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003a56: c5 fc 11 83 e0 00 00 00     	vmovups	%ymm0, 224(%rbx)
100003a5e: 83 bb d4 00 00 00 00        	cmpl	$0, 212(%rbx)
100003a65: 7e 1f                       	jle	31 <_main+0x1626>
100003a67: 48 8b 83 10 01 00 00        	movq	272(%rbx), %rax
100003a6e: 31 c9                       	xorl	%ecx, %ecx
100003a70: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003a77: 48 ff c1                    	incq	%rcx
100003a7a: 48 63 93 d4 00 00 00        	movslq	212(%rbx), %rdx
100003a81: 48 39 d1                    	cmpq	%rdx, %rcx
100003a84: 7c ea                       	jl	-22 <_main+0x1610>
100003a86: 48 8b bb 18 01 00 00        	movq	280(%rbx), %rdi
100003a8d: 48 8d 83 20 01 00 00        	leaq	288(%rbx), %rax
100003a94: 48 39 c7                    	cmpq	%rax, %rdi
100003a97: 74 08                       	je	8 <_main+0x1641>
100003a99: c5 f8 77                    	vzeroupper
100003a9c: e8 cb 33 00 00              	callq	13259 <dyld_stub_binder+0x100006e6c>
100003aa1: 48 8b 83 a8 00 00 00        	movq	168(%rbx), %rax
100003aa8: 48 85 c0                    	testq	%rax, %rax
100003aab: 74 12                       	je	18 <_main+0x165f>
100003aad: f0                          	lock
100003aae: ff 48 14                    	decl	20(%rax)
100003ab1: 75 0c                       	jne	12 <_main+0x165f>
100003ab3: 48 8d 7b 70                 	leaq	112(%rbx), %rdi
100003ab7: c5 f8 77                    	vzeroupper
100003aba: e8 77 33 00 00              	callq	13175 <dyld_stub_binder+0x100006e36>
100003abf: 48 c7 83 a8 00 00 00 00 00 00 00    	movq	$0, 168(%rbx)
100003aca: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003ace: c5 fc 11 83 80 00 00 00     	vmovups	%ymm0, 128(%rbx)
100003ad6: 83 7b 74 00                 	cmpl	$0, 116(%rbx)
100003ada: 7e 27                       	jle	39 <_main+0x16a3>
100003adc: 48 8b 83 b0 00 00 00        	movq	176(%rbx), %rax
100003ae3: 31 c9                       	xorl	%ecx, %ecx
100003ae5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003aef: 90                          	nop
100003af0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003af7: 48 ff c1                    	incq	%rcx
100003afa: 48 63 53 74                 	movslq	116(%rbx), %rdx
100003afe: 48 39 d1                    	cmpq	%rdx, %rcx
100003b01: 7c ed                       	jl	-19 <_main+0x1690>
100003b03: 48 8b bb b8 00 00 00        	movq	184(%rbx), %rdi
100003b0a: 48 8d 83 c0 00 00 00        	leaq	192(%rbx), %rax
100003b11: 48 39 c7                    	cmpq	%rax, %rdi
100003b14: 74 08                       	je	8 <_main+0x16be>
100003b16: c5 f8 77                    	vzeroupper
100003b19: e8 4e 33 00 00              	callq	13134 <dyld_stub_binder+0x100006e6c>
100003b1e: 48 8b 43 48                 	movq	72(%rbx), %rax
100003b22: 48 85 c0                    	testq	%rax, %rax
100003b25: 74 12                       	je	18 <_main+0x16d9>
100003b27: f0                          	lock
100003b28: ff 48 14                    	decl	20(%rax)
100003b2b: 75 0c                       	jne	12 <_main+0x16d9>
100003b2d: 48 8d 7b 10                 	leaq	16(%rbx), %rdi
100003b31: c5 f8 77                    	vzeroupper
100003b34: e8 fd 32 00 00              	callq	13053 <dyld_stub_binder+0x100006e36>
100003b39: 48 c7 43 48 00 00 00 00     	movq	$0, 72(%rbx)
100003b41: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003b45: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
100003b4a: 83 7b 14 00                 	cmpl	$0, 20(%rbx)
100003b4e: 7e 23                       	jle	35 <_main+0x1713>
100003b50: 48 8b 43 50                 	movq	80(%rbx), %rax
100003b54: 31 c9                       	xorl	%ecx, %ecx
100003b56: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003b60: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003b67: 48 ff c1                    	incq	%rcx
100003b6a: 48 63 53 14                 	movslq	20(%rbx), %rdx
100003b6e: 48 39 d1                    	cmpq	%rdx, %rcx
100003b71: 7c ed                       	jl	-19 <_main+0x1700>
100003b73: 48 8b 7b 58                 	movq	88(%rbx), %rdi
100003b77: 48 83 c3 60                 	addq	$96, %rbx
100003b7b: 48 39 df                    	cmpq	%rbx, %rdi
100003b7e: 74 08                       	je	8 <_main+0x1728>
100003b80: c5 f8 77                    	vzeroupper
100003b83: e8 e4 32 00 00              	callq	13028 <dyld_stub_binder+0x100006e6c>
100003b88: 48 83 c4 08                 	addq	$8, %rsp
100003b8c: 5b                          	popq	%rbx
100003b8d: 5d                          	popq	%rbp
100003b8e: c5 f8 77                    	vzeroupper
100003b91: c3                          	retq
100003b92: 48 89 c7                    	movq	%rax, %rdi
100003b95: e8 76 fe ff ff              	callq	-394 <_main+0x15b0>
100003b9a: 48 89 c7                    	movq	%rax, %rdi
100003b9d: e8 6e fe ff ff              	callq	-402 <_main+0x15b0>
100003ba2: 48 89 c7                    	movq	%rax, %rdi
100003ba5: e8 66 fe ff ff              	callq	-410 <_main+0x15b0>
100003baa: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100003bb0: 55                          	pushq	%rbp
100003bb1: 48 89 e5                    	movq	%rsp, %rbp
100003bb4: 53                          	pushq	%rbx
100003bb5: 50                          	pushq	%rax
100003bb6: 48 89 fb                    	movq	%rdi, %rbx
100003bb9: 48 8b 05 90 54 00 00        	movq	21648(%rip), %rax
100003bc0: 48 83 c0 10                 	addq	$16, %rax
100003bc4: 48 89 07                    	movq	%rax, (%rdi)
100003bc7: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100003bcb: 48 85 ff                    	testq	%rdi, %rdi
100003bce: 74 05                       	je	5 <_main+0x1775>
100003bd0: e8 fd 32 00 00              	callq	13053 <dyld_stub_binder+0x100006ed2>
100003bd5: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100003bd9: 48 83 c4 08                 	addq	$8, %rsp
100003bdd: 48 85 ff                    	testq	%rdi, %rdi
100003be0: 74 07                       	je	7 <_main+0x1789>
100003be2: 5b                          	popq	%rbx
100003be3: 5d                          	popq	%rbp
100003be4: e9 e9 32 00 00              	jmp	13033 <dyld_stub_binder+0x100006ed2>
100003be9: 5b                          	popq	%rbx
100003bea: 5d                          	popq	%rbp
100003beb: c3                          	retq
100003bec: 0f 1f 40 00                 	nopl	(%rax)
100003bf0: 55                          	pushq	%rbp
100003bf1: 48 89 e5                    	movq	%rsp, %rbp
100003bf4: 53                          	pushq	%rbx
100003bf5: 50                          	pushq	%rax
100003bf6: 48 89 fb                    	movq	%rdi, %rbx
100003bf9: 48 8b 05 50 54 00 00        	movq	21584(%rip), %rax
100003c00: 48 83 c0 10                 	addq	$16, %rax
100003c04: 48 89 07                    	movq	%rax, (%rdi)
100003c07: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100003c0b: 48 85 ff                    	testq	%rdi, %rdi
100003c0e: 74 05                       	je	5 <_main+0x17b5>
100003c10: e8 bd 32 00 00              	callq	12989 <dyld_stub_binder+0x100006ed2>
100003c15: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100003c19: 48 85 ff                    	testq	%rdi, %rdi
100003c1c: 74 05                       	je	5 <_main+0x17c3>
100003c1e: e8 af 32 00 00              	callq	12975 <dyld_stub_binder+0x100006ed2>
100003c23: 48 89 df                    	movq	%rbx, %rdi
100003c26: 48 83 c4 08                 	addq	$8, %rsp
100003c2a: 5b                          	popq	%rbx
100003c2b: 5d                          	popq	%rbp
100003c2c: e9 a1 32 00 00              	jmp	12961 <dyld_stub_binder+0x100006ed2>
100003c31: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003c3b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003c40: 55                          	pushq	%rbp
100003c41: 48 89 e5                    	movq	%rsp, %rbp
100003c44: 5d                          	popq	%rbp
100003c45: c3                          	retq
100003c46: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003c50: 55                          	pushq	%rbp
100003c51: 48 89 e5                    	movq	%rsp, %rbp
100003c54: 41 57                       	pushq	%r15
100003c56: 41 56                       	pushq	%r14
100003c58: 41 55                       	pushq	%r13
100003c5a: 41 54                       	pushq	%r12
100003c5c: 53                          	pushq	%rbx
100003c5d: 48 83 ec 28                 	subq	$40, %rsp
100003c61: 49 89 d6                    	movq	%rdx, %r14
100003c64: 49 89 f7                    	movq	%rsi, %r15
100003c67: 48 89 fb                    	movq	%rdi, %rbx
100003c6a: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100003c6e: 48 89 de                    	movq	%rbx, %rsi
100003c71: e8 26 32 00 00              	callq	12838 <dyld_stub_binder+0x100006e9c>
100003c76: 80 7d b0 00                 	cmpb	$0, -80(%rbp)
100003c7a: 0f 84 ae 00 00 00           	je	174 <_main+0x18ce>
100003c80: 48 8b 03                    	movq	(%rbx), %rax
100003c83: 48 8b 40 e8                 	movq	-24(%rax), %rax
100003c87: 4c 8d 24 03                 	leaq	(%rbx,%rax), %r12
100003c8b: 48 8b 7c 03 28              	movq	40(%rbx,%rax), %rdi
100003c90: 44 8b 6c 03 08              	movl	8(%rbx,%rax), %r13d
100003c95: 8b 84 03 90 00 00 00        	movl	144(%rbx,%rax), %eax
100003c9c: 83 f8 ff                    	cmpl	$-1, %eax
100003c9f: 75 4a                       	jne	74 <_main+0x188b>
100003ca1: 48 89 7d c0                 	movq	%rdi, -64(%rbp)
100003ca5: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100003ca9: 4c 89 e6                    	movq	%r12, %rsi
100003cac: e8 d3 31 00 00              	callq	12755 <dyld_stub_binder+0x100006e84>
100003cb1: 48 8b 35 90 53 00 00        	movq	21392(%rip), %rsi
100003cb8: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100003cbc: e8 bd 31 00 00              	callq	12733 <dyld_stub_binder+0x100006e7e>
100003cc1: 48 8b 08                    	movq	(%rax), %rcx
100003cc4: 48 89 c7                    	movq	%rax, %rdi
100003cc7: be 20 00 00 00              	movl	$32, %esi
100003ccc: ff 51 38                    	callq	*56(%rcx)
100003ccf: 88 45 d7                    	movb	%al, -41(%rbp)
100003cd2: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100003cd6: e8 d3 31 00 00              	callq	12755 <dyld_stub_binder+0x100006eae>
100003cdb: 0f be 45 d7                 	movsbl	-41(%rbp), %eax
100003cdf: 41 89 84 24 90 00 00 00     	movl	%eax, 144(%r12)
100003ce7: 48 8b 7d c0                 	movq	-64(%rbp), %rdi
100003ceb: 4d 01 fe                    	addq	%r15, %r14
100003cee: 41 81 e5 b0 00 00 00        	andl	$176, %r13d
100003cf5: 41 83 fd 20                 	cmpl	$32, %r13d
100003cf9: 4c 89 fa                    	movq	%r15, %rdx
100003cfc: 49 0f 44 d6                 	cmoveq	%r14, %rdx
100003d00: 44 0f be c8                 	movsbl	%al, %r9d
100003d04: 4c 89 fe                    	movq	%r15, %rsi
100003d07: 4c 89 f1                    	movq	%r14, %rcx
100003d0a: 4d 89 e0                    	movq	%r12, %r8
100003d0d: e8 9e 00 00 00              	callq	158 <_main+0x1950>
100003d12: 48 85 c0                    	testq	%rax, %rax
100003d15: 75 17                       	jne	23 <_main+0x18ce>
100003d17: 48 8b 03                    	movq	(%rbx), %rax
100003d1a: 48 8b 40 e8                 	movq	-24(%rax), %rax
100003d1e: 48 8d 3c 03                 	leaq	(%rbx,%rax), %rdi
100003d22: 8b 74 03 20                 	movl	32(%rbx,%rax), %esi
100003d26: 83 ce 05                    	orl	$5, %esi
100003d29: e8 8c 31 00 00              	callq	12684 <dyld_stub_binder+0x100006eba>
100003d2e: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100003d32: e8 6b 31 00 00              	callq	12651 <dyld_stub_binder+0x100006ea2>
100003d37: 48 89 d8                    	movq	%rbx, %rax
100003d3a: 48 83 c4 28                 	addq	$40, %rsp
100003d3e: 5b                          	popq	%rbx
100003d3f: 41 5c                       	popq	%r12
100003d41: 41 5d                       	popq	%r13
100003d43: 41 5e                       	popq	%r14
100003d45: 41 5f                       	popq	%r15
100003d47: 5d                          	popq	%rbp
100003d48: c3                          	retq
100003d49: eb 0e                       	jmp	14 <_main+0x18f9>
100003d4b: 49 89 c6                    	movq	%rax, %r14
100003d4e: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100003d52: e8 57 31 00 00              	callq	12631 <dyld_stub_binder+0x100006eae>
100003d57: eb 03                       	jmp	3 <_main+0x18fc>
100003d59: 49 89 c6                    	movq	%rax, %r14
100003d5c: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100003d60: e8 3d 31 00 00              	callq	12605 <dyld_stub_binder+0x100006ea2>
100003d65: eb 03                       	jmp	3 <_main+0x190a>
100003d67: 49 89 c6                    	movq	%rax, %r14
100003d6a: 4c 89 f7                    	movq	%r14, %rdi
100003d6d: e8 72 31 00 00              	callq	12658 <dyld_stub_binder+0x100006ee4>
100003d72: 48 8b 03                    	movq	(%rbx), %rax
100003d75: 48 8b 78 e8                 	movq	-24(%rax), %rdi
100003d79: 48 01 df                    	addq	%rbx, %rdi
100003d7c: e8 33 31 00 00              	callq	12595 <dyld_stub_binder+0x100006eb4>
100003d81: e8 64 31 00 00              	callq	12644 <dyld_stub_binder+0x100006eea>
100003d86: eb af                       	jmp	-81 <_main+0x18d7>
100003d88: 48 89 c3                    	movq	%rax, %rbx
100003d8b: e8 5a 31 00 00              	callq	12634 <dyld_stub_binder+0x100006eea>
100003d90: 48 89 df                    	movq	%rbx, %rdi
100003d93: e8 80 30 00 00              	callq	12416 <dyld_stub_binder+0x100006e18>
100003d98: 0f 0b                       	ud2
100003d9a: 48 89 c7                    	movq	%rax, %rdi
100003d9d: e8 6e fc ff ff              	callq	-914 <_main+0x15b0>
100003da2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003dac: 0f 1f 40 00                 	nopl	(%rax)
100003db0: 55                          	pushq	%rbp
100003db1: 48 89 e5                    	movq	%rsp, %rbp
100003db4: 41 57                       	pushq	%r15
100003db6: 41 56                       	pushq	%r14
100003db8: 41 55                       	pushq	%r13
100003dba: 41 54                       	pushq	%r12
100003dbc: 53                          	pushq	%rbx
100003dbd: 48 83 ec 38                 	subq	$56, %rsp
100003dc1: 48 85 ff                    	testq	%rdi, %rdi
100003dc4: 0f 84 17 01 00 00           	je	279 <_main+0x1a81>
100003dca: 4d 89 c4                    	movq	%r8, %r12
100003dcd: 49 89 cf                    	movq	%rcx, %r15
100003dd0: 49 89 fe                    	movq	%rdi, %r14
100003dd3: 44 89 4d bc                 	movl	%r9d, -68(%rbp)
100003dd7: 48 89 c8                    	movq	%rcx, %rax
100003dda: 48 29 f0                    	subq	%rsi, %rax
100003ddd: 49 8b 48 18                 	movq	24(%r8), %rcx
100003de1: 45 31 ed                    	xorl	%r13d, %r13d
100003de4: 48 29 c1                    	subq	%rax, %rcx
100003de7: 4c 0f 4f e9                 	cmovgq	%rcx, %r13
100003deb: 48 89 55 a8                 	movq	%rdx, -88(%rbp)
100003def: 48 89 d3                    	movq	%rdx, %rbx
100003df2: 48 29 f3                    	subq	%rsi, %rbx
100003df5: 48 85 db                    	testq	%rbx, %rbx
100003df8: 7e 15                       	jle	21 <_main+0x19af>
100003dfa: 49 8b 06                    	movq	(%r14), %rax
100003dfd: 4c 89 f7                    	movq	%r14, %rdi
100003e00: 48 89 da                    	movq	%rbx, %rdx
100003e03: ff 50 60                    	callq	*96(%rax)
100003e06: 48 39 d8                    	cmpq	%rbx, %rax
100003e09: 0f 85 d2 00 00 00           	jne	210 <_main+0x1a81>
100003e0f: 4d 85 ed                    	testq	%r13, %r13
100003e12: 0f 8e a1 00 00 00           	jle	161 <_main+0x1a59>
100003e18: 4c 89 65 b0                 	movq	%r12, -80(%rbp)
100003e1c: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003e20: c5 f8 29 45 c0              	vmovaps	%xmm0, -64(%rbp)
100003e25: 48 c7 45 d0 00 00 00 00     	movq	$0, -48(%rbp)
100003e2d: 49 83 fd 17                 	cmpq	$23, %r13
100003e31: 73 12                       	jae	18 <_main+0x19e5>
100003e33: 43 8d 44 2d 00              	leal	(%r13,%r13), %eax
100003e38: 88 45 c0                    	movb	%al, -64(%rbp)
100003e3b: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
100003e3f: 4c 8d 65 c1                 	leaq	-63(%rbp), %r12
100003e43: eb 27                       	jmp	39 <_main+0x1a0c>
100003e45: 49 8d 5d 10                 	leaq	16(%r13), %rbx
100003e49: 48 83 e3 f0                 	andq	$-16, %rbx
100003e4d: 48 89 df                    	movq	%rbx, %rdi
100003e50: e8 89 30 00 00              	callq	12425 <dyld_stub_binder+0x100006ede>
100003e55: 49 89 c4                    	movq	%rax, %r12
100003e58: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100003e5c: 48 83 cb 01                 	orq	$1, %rbx
100003e60: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
100003e64: 4c 89 6d c8                 	movq	%r13, -56(%rbp)
100003e68: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
100003e6c: 0f b6 75 bc                 	movzbl	-68(%rbp), %esi
100003e70: 4c 89 e7                    	movq	%r12, %rdi
100003e73: 4c 89 ea                    	movq	%r13, %rdx
100003e76: e8 7b 30 00 00              	callq	12411 <dyld_stub_binder+0x100006ef6>
100003e7b: 43 c6 04 2c 00              	movb	$0, (%r12,%r13)
100003e80: f6 45 c0 01                 	testb	$1, -64(%rbp)
100003e84: 74 06                       	je	6 <_main+0x1a2c>
100003e86: 48 8b 5d d0                 	movq	-48(%rbp), %rbx
100003e8a: eb 03                       	jmp	3 <_main+0x1a2f>
100003e8c: 48 ff c3                    	incq	%rbx
100003e8f: 4c 8b 65 b0                 	movq	-80(%rbp), %r12
100003e93: 49 8b 06                    	movq	(%r14), %rax
100003e96: 4c 89 f7                    	movq	%r14, %rdi
100003e99: 48 89 de                    	movq	%rbx, %rsi
100003e9c: 4c 89 ea                    	movq	%r13, %rdx
100003e9f: ff 50 60                    	callq	*96(%rax)
100003ea2: 48 89 c3                    	movq	%rax, %rbx
100003ea5: f6 45 c0 01                 	testb	$1, -64(%rbp)
100003ea9: 74 09                       	je	9 <_main+0x1a54>
100003eab: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
100003eaf: e8 1e 30 00 00              	callq	12318 <dyld_stub_binder+0x100006ed2>
100003eb4: 4c 39 eb                    	cmpq	%r13, %rbx
100003eb7: 75 28                       	jne	40 <_main+0x1a81>
100003eb9: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100003ebd: 49 29 f7                    	subq	%rsi, %r15
100003ec0: 4d 85 ff                    	testq	%r15, %r15
100003ec3: 7e 11                       	jle	17 <_main+0x1a76>
100003ec5: 49 8b 06                    	movq	(%r14), %rax
100003ec8: 4c 89 f7                    	movq	%r14, %rdi
100003ecb: 4c 89 fa                    	movq	%r15, %rdx
100003ece: ff 50 60                    	callq	*96(%rax)
100003ed1: 4c 39 f8                    	cmpq	%r15, %rax
100003ed4: 75 0b                       	jne	11 <_main+0x1a81>
100003ed6: 49 c7 44 24 18 00 00 00 00  	movq	$0, 24(%r12)
100003edf: eb 03                       	jmp	3 <_main+0x1a84>
100003ee1: 45 31 f6                    	xorl	%r14d, %r14d
100003ee4: 4c 89 f0                    	movq	%r14, %rax
100003ee7: 48 83 c4 38                 	addq	$56, %rsp
100003eeb: 5b                          	popq	%rbx
100003eec: 41 5c                       	popq	%r12
100003eee: 41 5d                       	popq	%r13
100003ef0: 41 5e                       	popq	%r14
100003ef2: 41 5f                       	popq	%r15
100003ef4: 5d                          	popq	%rbp
100003ef5: c3                          	retq
100003ef6: 48 89 c3                    	movq	%rax, %rbx
100003ef9: f6 45 c0 01                 	testb	$1, -64(%rbp)
100003efd: 74 09                       	je	9 <_main+0x1aa8>
100003eff: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
100003f03: e8 ca 2f 00 00              	callq	12234 <dyld_stub_binder+0x100006ed2>
100003f08: 48 89 df                    	movq	%rbx, %rdi
100003f0b: e8 08 2f 00 00              	callq	12040 <dyld_stub_binder+0x100006e18>
100003f10: 0f 0b                       	ud2
100003f12: 90                          	nop
100003f13: 90                          	nop
100003f14: 90                          	nop
100003f15: 90                          	nop
100003f16: 90                          	nop
100003f17: 90                          	nop
100003f18: 90                          	nop
100003f19: 90                          	nop
100003f1a: 90                          	nop
100003f1b: 90                          	nop
100003f1c: 90                          	nop
100003f1d: 90                          	nop
100003f1e: 90                          	nop
100003f1f: 90                          	nop
100003f20: 55                          	pushq	%rbp
100003f21: 48 89 e5                    	movq	%rsp, %rbp
100003f24: 48 8b 05 d5 50 00 00        	movq	20693(%rip), %rax
100003f2b: 80 38 00                    	cmpb	$0, (%rax)
100003f2e: 74 02                       	je	2 <_main+0x1ad2>
100003f30: 5d                          	popq	%rbp
100003f31: c3                          	retq
100003f32: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003f39: 5d                          	popq	%rbp
100003f3a: c3                          	retq
100003f3b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003f40: 55                          	pushq	%rbp
100003f41: 48 89 e5                    	movq	%rsp, %rbp
100003f44: 48 8b 05 d5 50 00 00        	movq	20693(%rip), %rax
100003f4b: 80 38 00                    	cmpb	$0, (%rax)
100003f4e: 74 02                       	je	2 <_main+0x1af2>
100003f50: 5d                          	popq	%rbp
100003f51: c3                          	retq
100003f52: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003f59: 5d                          	popq	%rbp
100003f5a: c3                          	retq
100003f5b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003f60: 55                          	pushq	%rbp
100003f61: 48 89 e5                    	movq	%rsp, %rbp
100003f64: 48 8b 05 cd 50 00 00        	movq	20685(%rip), %rax
100003f6b: 80 38 00                    	cmpb	$0, (%rax)
100003f6e: 74 02                       	je	2 <_main+0x1b12>
100003f70: 5d                          	popq	%rbp
100003f71: c3                          	retq
100003f72: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003f79: 5d                          	popq	%rbp
100003f7a: c3                          	retq
100003f7b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003f80: 55                          	pushq	%rbp
100003f81: 48 89 e5                    	movq	%rsp, %rbp
100003f84: 48 8b 05 a5 50 00 00        	movq	20645(%rip), %rax
100003f8b: 80 38 00                    	cmpb	$0, (%rax)
100003f8e: 74 02                       	je	2 <_main+0x1b32>
100003f90: 5d                          	popq	%rbp
100003f91: c3                          	retq
100003f92: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003f99: 5d                          	popq	%rbp
100003f9a: c3                          	retq
100003f9b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003fa0: 55                          	pushq	%rbp
100003fa1: 48 89 e5                    	movq	%rsp, %rbp
100003fa4: 48 8b 05 7d 50 00 00        	movq	20605(%rip), %rax
100003fab: 80 38 00                    	cmpb	$0, (%rax)
100003fae: 74 02                       	je	2 <_main+0x1b52>
100003fb0: 5d                          	popq	%rbp
100003fb1: c3                          	retq
100003fb2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003fb9: 5d                          	popq	%rbp
100003fba: c3                          	retq
100003fbb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003fc0: 55                          	pushq	%rbp
100003fc1: 48 89 e5                    	movq	%rsp, %rbp
100003fc4: 48 8b 05 3d 50 00 00        	movq	20541(%rip), %rax
100003fcb: 80 38 00                    	cmpb	$0, (%rax)
100003fce: 74 02                       	je	2 <_main+0x1b72>
100003fd0: 5d                          	popq	%rbp
100003fd1: c3                          	retq
100003fd2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003fd9: 5d                          	popq	%rbp
100003fda: c3                          	retq
100003fdb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003fe0: 55                          	pushq	%rbp
100003fe1: 48 89 e5                    	movq	%rsp, %rbp
100003fe4: 48 8b 05 25 50 00 00        	movq	20517(%rip), %rax
100003feb: 80 38 00                    	cmpb	$0, (%rax)
100003fee: 74 02                       	je	2 <_main+0x1b92>
100003ff0: 5d                          	popq	%rbp
100003ff1: c3                          	retq
100003ff2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003ff9: 5d                          	popq	%rbp
100003ffa: c3                          	retq
100003ffb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004000: 55                          	pushq	%rbp
100004001: 48 89 e5                    	movq	%rsp, %rbp
100004004: 48 8b 05 0d 50 00 00        	movq	20493(%rip), %rax
10000400b: 80 38 00                    	cmpb	$0, (%rax)
10000400e: 74 02                       	je	2 <_main+0x1bb2>
100004010: 5d                          	popq	%rbp
100004011: c3                          	retq
100004012: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004019: 5d                          	popq	%rbp
10000401a: c3                          	retq
10000401b: 90                          	nop
10000401c: 90                          	nop
10000401d: 90                          	nop
10000401e: 90                          	nop
10000401f: 90                          	nop

0000000100004020 __ZN11LineNetworkC2Ev:
100004020: 55                          	pushq	%rbp
100004021: 48 89 e5                    	movq	%rsp, %rbp
100004024: 41 57                       	pushq	%r15
100004026: 41 56                       	pushq	%r14
100004028: 41 54                       	pushq	%r12
10000402a: 53                          	pushq	%rbx
10000402b: 49 89 fc                    	movq	%rdi, %r12
10000402e: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004032: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
100004037: 48 8d 05 ba 50 00 00        	leaq	20666(%rip), %rax
10000403e: 48 89 07                    	movq	%rax, (%rdi)
100004041: c6 47 24 00                 	movb	$0, 36(%rdi)
100004045: bf 08 f0 07 00              	movl	$520200, %edi
10000404a: e8 89 2e 00 00              	callq	11913 <dyld_stub_binder+0x100006ed8>
10000404f: 49 89 c7                    	movq	%rax, %r15
100004052: 49 8d 5c 24 28              	leaq	40(%r12), %rbx
100004057: 48 89 03                    	movq	%rax, (%rbx)
10000405a: bf 08 f0 07 00              	movl	$520200, %edi
10000405f: e8 74 2e 00 00              	callq	11892 <dyld_stub_binder+0x100006ed8>
100004064: 49 89 44 24 30              	movq	%rax, 48(%r12)
100004069: 66 41 c7 07 00 00           	movw	$0, (%r15)
10000406f: 31 c0                       	xorl	%eax, %eax
100004071: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000407b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004080: 48 8b 0b                    	movq	(%rbx), %rcx
100004083: c6 44 01 02 00              	movb	$0, 2(%rcx,%rax)
100004088: 48 8b 0b                    	movq	(%rbx), %rcx
10000408b: c6 44 01 03 00              	movb	$0, 3(%rcx,%rax)
100004090: 48 8b 0b                    	movq	(%rbx), %rcx
100004093: c6 44 01 04 00              	movb	$0, 4(%rcx,%rax)
100004098: 48 8b 0b                    	movq	(%rbx), %rcx
10000409b: c6 44 01 05 00              	movb	$0, 5(%rcx,%rax)
1000040a0: 48 8b 0b                    	movq	(%rbx), %rcx
1000040a3: c6 44 01 06 00              	movb	$0, 6(%rcx,%rax)
1000040a8: 48 8b 0b                    	movq	(%rbx), %rcx
1000040ab: c6 44 01 07 00              	movb	$0, 7(%rcx,%rax)
1000040b0: 48 8b 0b                    	movq	(%rbx), %rcx
1000040b3: c6 44 01 08 00              	movb	$0, 8(%rcx,%rax)
1000040b8: 48 83 c0 07                 	addq	$7, %rax
1000040bc: 48 3d 06 f0 07 00           	cmpq	$520198, %rax
1000040c2: 75 bc                       	jne	-68 <__ZN11LineNetworkC2Ev+0x60>
1000040c4: 31 c0                       	xorl	%eax, %eax
1000040c6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000040d0: 49 8b 4c 24 30              	movq	48(%r12), %rcx
1000040d5: c6 04 01 00                 	movb	$0, (%rcx,%rax)
1000040d9: 49 8b 4c 24 30              	movq	48(%r12), %rcx
1000040de: c6 44 01 01 00              	movb	$0, 1(%rcx,%rax)
1000040e3: 49 8b 4c 24 30              	movq	48(%r12), %rcx
1000040e8: c6 44 01 02 00              	movb	$0, 2(%rcx,%rax)
1000040ed: 49 8b 4c 24 30              	movq	48(%r12), %rcx
1000040f2: c6 44 01 03 00              	movb	$0, 3(%rcx,%rax)
1000040f7: 49 8b 4c 24 30              	movq	48(%r12), %rcx
1000040fc: c6 44 01 04 00              	movb	$0, 4(%rcx,%rax)
100004101: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100004106: c6 44 01 05 00              	movb	$0, 5(%rcx,%rax)
10000410b: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100004110: c6 44 01 06 00              	movb	$0, 6(%rcx,%rax)
100004115: 49 8b 4c 24 30              	movq	48(%r12), %rcx
10000411a: c6 44 01 07 00              	movb	$0, 7(%rcx,%rax)
10000411f: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100004124: c6 44 01 08 00              	movb	$0, 8(%rcx,%rax)
100004129: 49 8b 4c 24 30              	movq	48(%r12), %rcx
10000412e: c6 44 01 09 00              	movb	$0, 9(%rcx,%rax)
100004133: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100004138: c6 44 01 0a 00              	movb	$0, 10(%rcx,%rax)
10000413d: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100004142: c6 44 01 0b 00              	movb	$0, 11(%rcx,%rax)
100004147: 48 83 c0 0c                 	addq	$12, %rax
10000414b: 48 3d 08 f0 07 00           	cmpq	$520200, %rax
100004151: 0f 85 79 ff ff ff           	jne	-135 <__ZN11LineNetworkC2Ev+0xb0>
100004157: 41 c7 44 24 20 29 be 97 01  	movl	$26721833, 32(%r12)
100004160: c5 f8 28 05 f8 2f 00 00     	vmovaps	12280(%rip), %xmm0
100004168: c4 c1 78 11 44 24 08        	vmovups	%xmm0, 8(%r12)
10000416f: 48 b8 1f 00 00 00 1f 00 00 00       	movabsq	$133143986207, %rax
100004179: 49 89 44 24 18              	movq	%rax, 24(%r12)
10000417e: 5b                          	popq	%rbx
10000417f: 41 5c                       	popq	%r12
100004181: 41 5e                       	popq	%r14
100004183: 41 5f                       	popq	%r15
100004185: 5d                          	popq	%rbp
100004186: c3                          	retq
100004187: 49 89 c6                    	movq	%rax, %r14
10000418a: 48 8b 05 bf 4e 00 00        	movq	20159(%rip), %rax
100004191: 48 83 c0 10                 	addq	$16, %rax
100004195: 49 89 04 24                 	movq	%rax, (%r12)
100004199: 4c 89 ff                    	movq	%r15, %rdi
10000419c: e8 31 2d 00 00              	callq	11569 <dyld_stub_binder+0x100006ed2>
1000041a1: 49 8b 7c 24 30              	movq	48(%r12), %rdi
1000041a6: 48 85 ff                    	testq	%rdi, %rdi
1000041a9: 74 21                       	je	33 <__ZN11LineNetworkC2Ev+0x1ac>
1000041ab: e8 22 2d 00 00              	callq	11554 <dyld_stub_binder+0x100006ed2>
1000041b0: 4c 89 f7                    	movq	%r14, %rdi
1000041b3: e8 60 2c 00 00              	callq	11360 <dyld_stub_binder+0x100006e18>
1000041b8: 0f 0b                       	ud2
1000041ba: 49 89 c6                    	movq	%rax, %r14
1000041bd: 48 8b 05 8c 4e 00 00        	movq	20108(%rip), %rax
1000041c4: 48 83 c0 10                 	addq	$16, %rax
1000041c8: 49 89 04 24                 	movq	%rax, (%r12)
1000041cc: 4c 89 f7                    	movq	%r14, %rdi
1000041cf: e8 44 2c 00 00              	callq	11332 <dyld_stub_binder+0x100006e18>
1000041d4: 0f 0b                       	ud2
1000041d6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

00000001000041e0 __ZN11LineNetworkC1Ev:
1000041e0: 55                          	pushq	%rbp
1000041e1: 48 89 e5                    	movq	%rsp, %rbp
1000041e4: 5d                          	popq	%rbp
1000041e5: e9 36 fe ff ff              	jmp	-458 <__ZN11LineNetworkC2Ev>
1000041ea: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)

00000001000041f0 __ZN11LineNetwork7forwardEv:
1000041f0: 55                          	pushq	%rbp
1000041f1: 48 89 e5                    	movq	%rsp, %rbp
1000041f4: 41 57                       	pushq	%r15
1000041f6: 41 56                       	pushq	%r14
1000041f8: 41 55                       	pushq	%r13
1000041fa: 41 54                       	pushq	%r12
1000041fc: 53                          	pushq	%rbx
1000041fd: 48 83 ec 48                 	subq	$72, %rsp
100004201: 49 89 fe                    	movq	%rdi, %r14
100004204: 0f b6 47 24                 	movzbl	36(%rdi), %eax
100004208: 31 c9                       	xorl	%ecx, %ecx
10000420a: 48 85 c0                    	testq	%rax, %rax
10000420d: 0f 94 c1                    	sete	%cl
100004210: 48 8b 7c cf 28              	movq	40(%rdi,%rcx,8), %rdi
100004215: 49 8b 74 c6 28              	movq	40(%r14,%rax,8), %rsi
10000421a: 48 8d 15 7f 31 00 00        	leaq	12671(%rip), %rdx
100004221: 48 8d 0d c0 31 00 00        	leaq	12736(%rip), %rcx
100004228: 41 b8 37 00 00 00           	movl	$55, %r8d
10000422e: e8 8d 1a 00 00              	callq	6797 <__ZN11LineNetwork7forwardEv+0x1ad0>
100004233: 41 0f b6 46 24              	movzbl	36(%r14), %eax
100004238: 48 83 f0 01                 	xorq	$1, %rax
10000423c: 41 88 46 24                 	movb	%al, 36(%r14)
100004240: 31 c9                       	xorl	%ecx, %ecx
100004242: 84 c0                       	testb	%al, %al
100004244: 0f 94 c1                    	sete	%cl
100004247: 49 8b 54 ce 28              	movq	40(%r14,%rcx,8), %rdx
10000424c: 49 8b 74 c6 28              	movq	40(%r14,%rax,8), %rsi
100004251: 48 8d 86 08 f0 07 00        	leaq	520200(%rsi), %rax
100004258: 48 39 c2                    	cmpq	%rax, %rdx
10000425b: 73 1c                       	jae	28 <__ZN11LineNetwork7forwardEv+0x89>
10000425d: 48 8d 82 08 f0 07 00        	leaq	520200(%rdx), %rax
100004264: bf 08 f0 07 00              	movl	$520200, %edi
100004269: 48 39 c6                    	cmpq	%rax, %rsi
10000426c: 73 0b                       	jae	11 <__ZN11LineNetwork7forwardEv+0x89>
10000426e: 48 89 f0                    	movq	%rsi, %rax
100004271: 48 89 d1                    	movq	%rdx, %rcx
100004274: e9 b1 04 00 00              	jmp	1201 <__ZN11LineNetwork7forwardEv+0x53a>
100004279: 48 8d 86 00 f0 07 00        	leaq	520192(%rsi), %rax
100004280: 48 8d 8a 00 f0 07 00        	leaq	520192(%rdx), %rcx
100004287: bf 20 00 00 00              	movl	$32, %edi
10000428c: 0f 1f 40 00                 	nopl	(%rax)
100004290: c5 7a 6f 74 be 80           	vmovdqu	-128(%rsi,%rdi,4), %xmm14
100004296: c5 7a 6f 7c be 90           	vmovdqu	-112(%rsi,%rdi,4), %xmm15
10000429c: c5 fa 6f 54 be a0           	vmovdqu	-96(%rsi,%rdi,4), %xmm2
1000042a2: c5 fa 6f 5c be b0           	vmovdqu	-80(%rsi,%rdi,4), %xmm3
1000042a8: c5 79 6f 1d c0 2e 00 00     	vmovdqa	11968(%rip), %xmm11
1000042b0: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
1000042b5: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
1000042ba: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
1000042be: c5 79 6f 05 ba 2e 00 00     	vmovdqa	11962(%rip), %xmm8
1000042c6: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
1000042cb: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
1000042d0: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
1000042d4: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
1000042da: c5 fa 6f 64 be f0           	vmovdqu	-16(%rsi,%rdi,4), %xmm4
1000042e0: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
1000042e5: c4 e3 fd 00 6c be e0 4e     	vpermq	$78, -32(%rsi,%rdi,4), %ymm5
1000042ed: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
1000042f3: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
1000042f8: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
1000042fc: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100004302: c5 fa 6f 74 be d0           	vmovdqu	-48(%rsi,%rdi,4), %xmm6
100004308: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
10000430d: c4 63 fd 00 4c be c0 4e     	vpermq	$78, -64(%rsi,%rdi,4), %ymm9
100004315: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
10000431b: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
100004320: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100004325: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000432b: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100004331: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004337: c5 79 6f 05 51 2e 00 00     	vmovdqa	11857(%rip), %xmm8
10000433f: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100004344: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100004349: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
10000434d: c5 79 6f 1d 4b 2e 00 00     	vmovdqa	11851(%rip), %xmm11
100004355: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
10000435a: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000435f: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
100004363: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
100004369: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
10000436e: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
100004373: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004377: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
10000437d: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
100004382: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
100004387: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
10000438b: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004391: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004397: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
10000439d: c5 79 6f 1d 0b 2e 00 00     	vmovdqa	11787(%rip), %xmm11
1000043a5: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
1000043aa: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
1000043af: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
1000043b3: c5 f9 6f 0d 05 2e 00 00     	vmovdqa	11781(%rip), %xmm1
1000043bb: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
1000043bf: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
1000043c4: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
1000043c9: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000043cd: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
1000043d3: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
1000043d8: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
1000043dd: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000043e1: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000043e7: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
1000043ec: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
1000043f1: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
1000043f5: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000043fb: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004401: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100004407: c5 f9 6f 0d c1 2d 00 00     	vmovdqa	11713(%rip), %xmm1
10000440f: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100004414: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100004419: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000441d: c5 f9 6f 15 bb 2d 00 00     	vmovdqa	11707(%rip), %xmm2
100004425: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100004429: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
10000442e: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100004433: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004437: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
10000443d: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100004442: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100004447: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000444b: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100004451: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
100004456: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
10000445b: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
10000445f: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100004465: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
10000446b: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
100004471: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
100004476: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
10000447b: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
100004480: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
100004485: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
10000448a: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
10000448e: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100004492: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100004496: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
10000449a: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
10000449e: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000044a2: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000044a6: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000044aa: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
1000044b0: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
1000044b6: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
1000044bc: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
1000044c2: c5 fe 7f 4c ba c0           	vmovdqu	%ymm1, -64(%rdx,%rdi,4)
1000044c8: c5 fe 7f 44 ba e0           	vmovdqu	%ymm0, -32(%rdx,%rdi,4)
1000044ce: c5 fe 7f 6c ba a0           	vmovdqu	%ymm5, -96(%rdx,%rdi,4)
1000044d4: c5 fe 7f 54 ba 80           	vmovdqu	%ymm2, -128(%rdx,%rdi,4)
1000044da: c5 7a 6f 24 be              	vmovdqu	(%rsi,%rdi,4), %xmm12
1000044df: c5 7a 6f 6c be 10           	vmovdqu	16(%rsi,%rdi,4), %xmm13
1000044e5: c5 7a 6f 74 be 20           	vmovdqu	32(%rsi,%rdi,4), %xmm14
1000044eb: c5 fa 6f 5c be 30           	vmovdqu	48(%rsi,%rdi,4), %xmm3
1000044f1: c5 f9 6f 05 77 2c 00 00     	vmovdqa	11383(%rip), %xmm0
1000044f9: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
1000044fe: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
100004503: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100004507: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000450b: c5 f9 6f 05 6d 2c 00 00     	vmovdqa	11373(%rip), %xmm0
100004513: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100004518: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
10000451d: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
100004521: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004525: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
10000452b: c5 fa 6f 64 be 70           	vmovdqu	112(%rsi,%rdi,4), %xmm4
100004531: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100004536: c4 e3 fd 00 6c be 60 4e     	vpermq	$78, 96(%rsi,%rdi,4), %ymm5
10000453e: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004544: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
100004549: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000454d: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
100004553: c5 fa 6f 74 be 50           	vmovdqu	80(%rsi,%rdi,4), %xmm6
100004559: c4 e3 fd 00 7c be 40 4e     	vpermq	$78, 64(%rsi,%rdi,4), %ymm7
100004561: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
100004566: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
10000456c: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
100004571: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
100004575: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000457b: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
100004581: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
100004587: c5 79 6f 3d 01 2c 00 00     	vmovdqa	11265(%rip), %xmm15
10000458f: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
100004594: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
100004599: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
10000459d: c5 f9 6f 05 fb 2b 00 00     	vmovdqa	11259(%rip), %xmm0
1000045a5: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000045aa: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000045af: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000045b3: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
1000045b9: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
1000045be: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
1000045c3: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000045c7: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000045cd: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
1000045d2: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
1000045d7: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
1000045db: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000045e1: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000045e7: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
1000045ed: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
1000045f2: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
1000045f7: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
1000045fb: c5 f9 6f 05 bd 2b 00 00     	vmovdqa	11197(%rip), %xmm0
100004603: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100004608: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
10000460d: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004611: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
100004617: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
10000461c: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100004621: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004625: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
10000462a: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
10000462f: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100004633: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004639: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000463f: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004645: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
10000464b: c5 79 6f 3d 7d 2b 00 00     	vmovdqa	11133(%rip), %xmm15
100004653: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
100004658: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
10000465d: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004661: c5 f9 6f 05 77 2b 00 00     	vmovdqa	11127(%rip), %xmm0
100004669: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
10000466e: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
100004673: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004677: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
10000467d: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
100004682: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
100004687: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000468b: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
100004690: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
100004695: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100004699: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
10000469f: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
1000046a5: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
1000046ab: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
1000046b1: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
1000046b6: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
1000046bb: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
1000046c0: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000046c5: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
1000046c9: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000046cd: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
1000046d1: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
1000046d5: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000046d9: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000046dd: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000046e1: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000046e5: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
1000046eb: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
1000046f1: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
1000046f7: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
1000046fd: c5 fe 7f 4c ba 40           	vmovdqu	%ymm1, 64(%rdx,%rdi,4)
100004703: c5 fe 7f 44 ba 60           	vmovdqu	%ymm0, 96(%rdx,%rdi,4)
100004709: c5 fe 7f 5c ba 20           	vmovdqu	%ymm3, 32(%rdx,%rdi,4)
10000470f: c5 fe 7f 14 ba              	vmovdqu	%ymm2, (%rdx,%rdi,4)
100004714: 48 83 c7 40                 	addq	$64, %rdi
100004718: 48 81 ff 20 fc 01 00        	cmpq	$130080, %rdi
10000471f: 0f 85 6b fb ff ff           	jne	-1173 <__ZN11LineNetwork7forwardEv+0xa0>
100004725: bf 08 00 00 00              	movl	$8, %edi
10000472a: 89 fa                       	movl	%edi, %edx
10000472c: 31 f6                       	xorl	%esi, %esi
10000472e: 31 ff                       	xorl	%edi, %edi
100004730: 0f b6 1c 38                 	movzbl	(%rax,%rdi), %ebx
100004734: 84 db                       	testb	%bl, %bl
100004736: 0f 48 de                    	cmovsl	%esi, %ebx
100004739: 88 1c 39                    	movb	%bl, (%rcx,%rdi)
10000473c: 0f b6 5c 38 01              	movzbl	1(%rax,%rdi), %ebx
100004741: 84 db                       	testb	%bl, %bl
100004743: 0f 48 de                    	cmovsl	%esi, %ebx
100004746: 88 5c 39 01                 	movb	%bl, 1(%rcx,%rdi)
10000474a: 0f b6 5c 38 02              	movzbl	2(%rax,%rdi), %ebx
10000474f: 84 db                       	testb	%bl, %bl
100004751: 0f 48 de                    	cmovsl	%esi, %ebx
100004754: 88 5c 39 02                 	movb	%bl, 2(%rcx,%rdi)
100004758: 0f b6 5c 38 03              	movzbl	3(%rax,%rdi), %ebx
10000475d: 84 db                       	testb	%bl, %bl
10000475f: 0f 48 de                    	cmovsl	%esi, %ebx
100004762: 88 5c 39 03                 	movb	%bl, 3(%rcx,%rdi)
100004766: 0f b6 5c 38 04              	movzbl	4(%rax,%rdi), %ebx
10000476b: 84 db                       	testb	%bl, %bl
10000476d: 0f 48 de                    	cmovsl	%esi, %ebx
100004770: 88 5c 39 04                 	movb	%bl, 4(%rcx,%rdi)
100004774: 0f b6 5c 38 05              	movzbl	5(%rax,%rdi), %ebx
100004779: 84 db                       	testb	%bl, %bl
10000477b: 0f 48 de                    	cmovsl	%esi, %ebx
10000477e: 88 5c 39 05                 	movb	%bl, 5(%rcx,%rdi)
100004782: 0f b6 5c 38 06              	movzbl	6(%rax,%rdi), %ebx
100004787: 84 db                       	testb	%bl, %bl
100004789: 0f 48 de                    	cmovsl	%esi, %ebx
10000478c: 88 5c 39 06                 	movb	%bl, 6(%rcx,%rdi)
100004790: 0f b6 5c 38 07              	movzbl	7(%rax,%rdi), %ebx
100004795: 84 db                       	testb	%bl, %bl
100004797: 0f 48 de                    	cmovsl	%esi, %ebx
10000479a: 88 5c 39 07                 	movb	%bl, 7(%rcx,%rdi)
10000479e: 48 83 c7 08                 	addq	$8, %rdi
1000047a2: 39 fa                       	cmpl	%edi, %edx
1000047a4: 75 8a                       	jne	-118 <__ZN11LineNetwork7forwardEv+0x540>
1000047a6: 41 0f b6 46 24              	movzbl	36(%r14), %eax
1000047ab: 48 83 f0 01                 	xorq	$1, %rax
1000047af: 41 88 46 24                 	movb	%al, 36(%r14)
1000047b3: 31 c9                       	xorl	%ecx, %ecx
1000047b5: 84 c0                       	testb	%al, %al
1000047b7: 0f 94 c1                    	sete	%cl
1000047ba: 4d 8b 64 ce 28              	movq	40(%r14,%rcx,8), %r12
1000047bf: 4c 89 75 90                 	movq	%r14, -112(%rbp)
1000047c3: 49 8b 44 c6 28              	movq	40(%r14,%rax,8), %rax
1000047c8: 48 89 45 c8                 	movq	%rax, -56(%rbp)
1000047cc: 31 c0                       	xorl	%eax, %eax
1000047ce: eb 18                       	jmp	24 <__ZN11LineNetwork7forwardEv+0x5f8>
1000047d0: 48 8b 45 d0                 	movq	-48(%rbp), %rax
1000047d4: 48 ff c0                    	incq	%rax
1000047d7: 4c 8b 65 c0                 	movq	-64(%rbp), %r12
1000047db: 49 ff c4                    	incq	%r12
1000047de: 48 83 f8 08                 	cmpq	$8, %rax
1000047e2: 0f 84 00 01 00 00           	je	256 <__ZN11LineNetwork7forwardEv+0x6f8>
1000047e8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
1000047ec: 48 8d 04 c0                 	leaq	(%rax,%rax,8), %rax
1000047f0: 48 8d 0d f9 2b 00 00        	leaq	11257(%rip), %rcx
1000047f7: 48 8d 14 c1                 	leaq	(%rcx,%rax,8), %rdx
1000047fb: 48 89 55 98                 	movq	%rdx, -104(%rbp)
1000047ff: 48 8d 54 c1 18              	leaq	24(%rcx,%rax,8), %rdx
100004804: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100004808: 48 8d 44 c1 30              	leaq	48(%rcx,%rax,8), %rax
10000480d: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100004811: 4c 89 65 c0                 	movq	%r12, -64(%rbp)
100004815: 48 8b 5d c8                 	movq	-56(%rbp), %rbx
100004819: 31 c0                       	xorl	%eax, %eax
10000481b: eb 22                       	jmp	34 <__ZN11LineNetwork7forwardEv+0x64f>
10000481d: 0f 1f 00                    	nopl	(%rax)
100004820: 48 8b 45 b8                 	movq	-72(%rbp), %rax
100004824: 48 ff c0                    	incq	%rax
100004827: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
10000482b: 48 81 c3 f0 0f 00 00        	addq	$4080, %rbx
100004832: 49 81 c4 f8 03 00 00        	addq	$1016, %r12
100004839: 48 83 f8 7f                 	cmpq	$127, %rax
10000483d: 74 91                       	je	-111 <__ZN11LineNetwork7forwardEv+0x5e0>
10000483f: 48 89 45 b8                 	movq	%rax, -72(%rbp)
100004843: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
100004847: 45 31 ed                    	xorl	%r13d, %r13d
10000484a: eb 15                       	jmp	21 <__ZN11LineNetwork7forwardEv+0x671>
10000484c: 0f 1f 40 00                 	nopl	(%rax)
100004850: 43 88 04 ec                 	movb	%al, (%r12,%r13,8)
100004854: 49 ff c5                    	incq	%r13
100004857: 48 83 c3 10                 	addq	$16, %rbx
10000485b: 49 83 fd 7f                 	cmpq	$127, %r13
10000485f: 74 bf                       	je	-65 <__ZN11LineNetwork7forwardEv+0x630>
100004861: 48 89 df                    	movq	%rbx, %rdi
100004864: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100004868: c5 f8 77                    	vzeroupper
10000486b: e8 f0 21 00 00              	callq	8688 <__ZN11LineNetwork7forwardEv+0x2870>
100004870: 41 89 c6                    	movl	%eax, %r14d
100004873: 48 8d bb f8 07 00 00        	leaq	2040(%rbx), %rdi
10000487a: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
10000487e: e8 dd 21 00 00              	callq	8669 <__ZN11LineNetwork7forwardEv+0x2870>
100004883: 41 89 c7                    	movl	%eax, %r15d
100004886: 45 01 f7                    	addl	%r14d, %r15d
100004889: 48 8d bb f0 0f 00 00        	leaq	4080(%rbx), %rdi
100004890: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100004894: e8 c7 21 00 00              	callq	8647 <__ZN11LineNetwork7forwardEv+0x2870>
100004899: 44 01 f8                    	addl	%r15d, %eax
10000489c: 48 8b 4d d0                 	movq	-48(%rbp), %rcx
1000048a0: 48 8d 15 89 2d 00 00        	leaq	11657(%rip), %rdx
1000048a7: 0f be 0c 11                 	movsbl	(%rcx,%rdx), %ecx
1000048ab: 01 c1                       	addl	%eax, %ecx
1000048ad: 6b c1 37                    	imull	$55, %ecx, %eax
1000048b0: 48 98                       	cltq
1000048b2: 48 69 c8 09 04 02 81        	imulq	$-2130574327, %rax, %rcx
1000048b9: 48 c1 e9 20                 	shrq	$32, %rcx
1000048bd: 01 c8                       	addl	%ecx, %eax
1000048bf: 89 c1                       	movl	%eax, %ecx
1000048c1: c1 e9 1f                    	shrl	$31, %ecx
1000048c4: c1 f8 0d                    	sarl	$13, %eax
1000048c7: 01 c8                       	addl	%ecx, %eax
1000048c9: 3d 80 00 00 00              	cmpl	$128, %eax
1000048ce: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x6e5>
1000048d0: b8 7f 00 00 00              	movl	$127, %eax
1000048d5: 83 f8 81                    	cmpl	$-127, %eax
1000048d8: 0f 8f 72 ff ff ff           	jg	-142 <__ZN11LineNetwork7forwardEv+0x660>
1000048de: b8 81 00 00 00              	movl	$129, %eax
1000048e3: e9 68 ff ff ff              	jmp	-152 <__ZN11LineNetwork7forwardEv+0x660>
1000048e8: 4c 8b 45 90                 	movq	-112(%rbp), %r8
1000048ec: 41 0f b6 40 24              	movzbl	36(%r8), %eax
1000048f1: 48 83 f0 01                 	xorq	$1, %rax
1000048f5: 41 88 40 24                 	movb	%al, 36(%r8)
1000048f9: 31 c9                       	xorl	%ecx, %ecx
1000048fb: 84 c0                       	testb	%al, %al
1000048fd: 0f 94 c1                    	sete	%cl
100004900: 49 8b 54 c8 28              	movq	40(%r8,%rcx,8), %rdx
100004905: 49 8b 74 c0 28              	movq	40(%r8,%rax,8), %rsi
10000490a: 48 8d 86 08 f8 01 00        	leaq	129032(%rsi), %rax
100004911: 48 39 c2                    	cmpq	%rax, %rdx
100004914: 73 1c                       	jae	28 <__ZN11LineNetwork7forwardEv+0x742>
100004916: 48 8d 82 08 f8 01 00        	leaq	129032(%rdx), %rax
10000491d: bf 08 f8 01 00              	movl	$129032, %edi
100004922: 48 39 c6                    	cmpq	%rax, %rsi
100004925: 73 0b                       	jae	11 <__ZN11LineNetwork7forwardEv+0x742>
100004927: 48 89 f0                    	movq	%rsi, %rax
10000492a: 48 89 d1                    	movq	%rdx, %rcx
10000492d: e9 b8 04 00 00              	jmp	1208 <__ZN11LineNetwork7forwardEv+0xbfa>
100004932: 48 8d 86 00 f8 01 00        	leaq	129024(%rsi), %rax
100004939: 48 8d 8a 00 f8 01 00        	leaq	129024(%rdx), %rcx
100004940: bf 20 00 00 00              	movl	$32, %edi
100004945: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000494f: 90                          	nop
100004950: c5 7a 6f 74 be 80           	vmovdqu	-128(%rsi,%rdi,4), %xmm14
100004956: c5 7a 6f 7c be 90           	vmovdqu	-112(%rsi,%rdi,4), %xmm15
10000495c: c5 fa 6f 54 be a0           	vmovdqu	-96(%rsi,%rdi,4), %xmm2
100004962: c5 fa 6f 5c be b0           	vmovdqu	-80(%rsi,%rdi,4), %xmm3
100004968: c5 79 6f 1d 00 28 00 00     	vmovdqa	10240(%rip), %xmm11
100004970: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100004975: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
10000497a: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000497e: c5 79 6f 05 fa 27 00 00     	vmovdqa	10234(%rip), %xmm8
100004986: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
10000498b: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
100004990: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004994: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
10000499a: c5 fa 6f 64 be f0           	vmovdqu	-16(%rsi,%rdi,4), %xmm4
1000049a0: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
1000049a5: c4 e3 fd 00 6c be e0 4e     	vpermq	$78, -32(%rsi,%rdi,4), %ymm5
1000049ad: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
1000049b3: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
1000049b8: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
1000049bc: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
1000049c2: c5 fa 6f 74 be d0           	vmovdqu	-48(%rsi,%rdi,4), %xmm6
1000049c8: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
1000049cd: c4 63 fd 00 4c be c0 4e     	vpermq	$78, -64(%rsi,%rdi,4), %ymm9
1000049d5: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
1000049db: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
1000049e0: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
1000049e5: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000049eb: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
1000049f1: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
1000049f7: c5 79 6f 05 91 27 00 00     	vmovdqa	10129(%rip), %xmm8
1000049ff: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100004a04: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100004a09: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
100004a0d: c5 79 6f 1d 8b 27 00 00     	vmovdqa	10123(%rip), %xmm11
100004a15: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
100004a1a: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100004a1f: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
100004a23: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
100004a29: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
100004a2e: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
100004a33: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004a37: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004a3d: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
100004a42: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
100004a47: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100004a4b: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004a51: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004a57: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
100004a5d: c5 79 6f 1d 4b 27 00 00     	vmovdqa	10059(%rip), %xmm11
100004a65: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100004a6a: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
100004a6f: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
100004a73: c5 f9 6f 0d 45 27 00 00     	vmovdqa	10053(%rip), %xmm1
100004a7b: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
100004a7f: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100004a84: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100004a89: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004a8d: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100004a93: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100004a98: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004a9d: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004aa1: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004aa7: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
100004aac: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100004ab1: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100004ab5: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004abb: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004ac1: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100004ac7: c5 f9 6f 0d 01 27 00 00     	vmovdqa	9985(%rip), %xmm1
100004acf: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100004ad4: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100004ad9: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
100004add: c5 f9 6f 15 fb 26 00 00     	vmovdqa	9979(%rip), %xmm2
100004ae5: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100004ae9: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
100004aee: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100004af3: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004af7: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
100004afd: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100004b02: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100004b07: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004b0b: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100004b11: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
100004b16: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
100004b1b: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100004b1f: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100004b25: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100004b2b: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
100004b31: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
100004b36: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
100004b3b: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
100004b40: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
100004b45: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100004b4a: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100004b4e: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100004b52: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100004b56: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100004b5a: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100004b5e: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100004b62: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100004b66: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100004b6a: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100004b70: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
100004b76: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100004b7c: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100004b82: c5 fe 7f 4c ba c0           	vmovdqu	%ymm1, -64(%rdx,%rdi,4)
100004b88: c5 fe 7f 44 ba e0           	vmovdqu	%ymm0, -32(%rdx,%rdi,4)
100004b8e: c5 fe 7f 6c ba a0           	vmovdqu	%ymm5, -96(%rdx,%rdi,4)
100004b94: c5 fe 7f 54 ba 80           	vmovdqu	%ymm2, -128(%rdx,%rdi,4)
100004b9a: c5 7a 6f 24 be              	vmovdqu	(%rsi,%rdi,4), %xmm12
100004b9f: c5 7a 6f 6c be 10           	vmovdqu	16(%rsi,%rdi,4), %xmm13
100004ba5: c5 7a 6f 74 be 20           	vmovdqu	32(%rsi,%rdi,4), %xmm14
100004bab: c5 fa 6f 5c be 30           	vmovdqu	48(%rsi,%rdi,4), %xmm3
100004bb1: c5 f9 6f 05 b7 25 00 00     	vmovdqa	9655(%rip), %xmm0
100004bb9: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
100004bbe: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
100004bc3: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100004bc7: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100004bcb: c5 f9 6f 05 ad 25 00 00     	vmovdqa	9645(%rip), %xmm0
100004bd3: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100004bd8: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
100004bdd: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
100004be1: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004be5: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
100004beb: c5 fa 6f 64 be 70           	vmovdqu	112(%rsi,%rdi,4), %xmm4
100004bf1: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100004bf6: c4 e3 fd 00 6c be 60 4e     	vpermq	$78, 96(%rsi,%rdi,4), %ymm5
100004bfe: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004c04: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
100004c09: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
100004c0d: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
100004c13: c5 fa 6f 74 be 50           	vmovdqu	80(%rsi,%rdi,4), %xmm6
100004c19: c4 e3 fd 00 7c be 40 4e     	vpermq	$78, 64(%rsi,%rdi,4), %ymm7
100004c21: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
100004c26: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
100004c2c: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
100004c31: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
100004c35: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004c3b: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
100004c41: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
100004c47: c5 79 6f 3d 41 25 00 00     	vmovdqa	9537(%rip), %xmm15
100004c4f: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
100004c54: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
100004c59: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
100004c5d: c5 f9 6f 05 3b 25 00 00     	vmovdqa	9531(%rip), %xmm0
100004c65: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100004c6a: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100004c6f: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004c73: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
100004c79: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
100004c7e: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
100004c83: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004c87: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004c8d: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100004c92: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100004c97: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100004c9b: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004ca1: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004ca7: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004cad: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100004cb2: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100004cb7: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100004cbb: c5 f9 6f 05 fd 24 00 00     	vmovdqa	9469(%rip), %xmm0
100004cc3: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100004cc8: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100004ccd: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004cd1: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
100004cd7: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100004cdc: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100004ce1: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004ce5: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100004cea: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100004cef: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100004cf3: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004cf9: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004cff: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004d05: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
100004d0b: c5 79 6f 3d bd 24 00 00     	vmovdqa	9405(%rip), %xmm15
100004d13: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
100004d18: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
100004d1d: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004d21: c5 f9 6f 05 b7 24 00 00     	vmovdqa	9399(%rip), %xmm0
100004d29: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
100004d2e: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
100004d33: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004d37: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
100004d3d: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
100004d42: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
100004d47: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004d4b: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
100004d50: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
100004d55: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100004d59: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100004d5f: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100004d65: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100004d6b: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
100004d71: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
100004d76: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
100004d7b: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
100004d80: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100004d85: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100004d89: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100004d8d: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100004d91: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100004d95: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100004d99: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100004d9d: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100004da1: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100004da5: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100004dab: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100004db1: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
100004db7: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100004dbd: c5 fe 7f 4c ba 40           	vmovdqu	%ymm1, 64(%rdx,%rdi,4)
100004dc3: c5 fe 7f 44 ba 60           	vmovdqu	%ymm0, 96(%rdx,%rdi,4)
100004dc9: c5 fe 7f 5c ba 20           	vmovdqu	%ymm3, 32(%rdx,%rdi,4)
100004dcf: c5 fe 7f 14 ba              	vmovdqu	%ymm2, (%rdx,%rdi,4)
100004dd4: 48 83 c7 40                 	addq	$64, %rdi
100004dd8: 48 81 ff 20 7e 00 00        	cmpq	$32288, %rdi
100004ddf: 0f 85 6b fb ff ff           	jne	-1173 <__ZN11LineNetwork7forwardEv+0x760>
100004de5: bf 08 00 00 00              	movl	$8, %edi
100004dea: 89 fa                       	movl	%edi, %edx
100004dec: 31 f6                       	xorl	%esi, %esi
100004dee: 31 ff                       	xorl	%edi, %edi
100004df0: 0f b6 1c 38                 	movzbl	(%rax,%rdi), %ebx
100004df4: 84 db                       	testb	%bl, %bl
100004df6: 0f 48 de                    	cmovsl	%esi, %ebx
100004df9: 88 1c 39                    	movb	%bl, (%rcx,%rdi)
100004dfc: 0f b6 5c 38 01              	movzbl	1(%rax,%rdi), %ebx
100004e01: 84 db                       	testb	%bl, %bl
100004e03: 0f 48 de                    	cmovsl	%esi, %ebx
100004e06: 88 5c 39 01                 	movb	%bl, 1(%rcx,%rdi)
100004e0a: 0f b6 5c 38 02              	movzbl	2(%rax,%rdi), %ebx
100004e0f: 84 db                       	testb	%bl, %bl
100004e11: 0f 48 de                    	cmovsl	%esi, %ebx
100004e14: 88 5c 39 02                 	movb	%bl, 2(%rcx,%rdi)
100004e18: 0f b6 5c 38 03              	movzbl	3(%rax,%rdi), %ebx
100004e1d: 84 db                       	testb	%bl, %bl
100004e1f: 0f 48 de                    	cmovsl	%esi, %ebx
100004e22: 88 5c 39 03                 	movb	%bl, 3(%rcx,%rdi)
100004e26: 0f b6 5c 38 04              	movzbl	4(%rax,%rdi), %ebx
100004e2b: 84 db                       	testb	%bl, %bl
100004e2d: 0f 48 de                    	cmovsl	%esi, %ebx
100004e30: 88 5c 39 04                 	movb	%bl, 4(%rcx,%rdi)
100004e34: 0f b6 5c 38 05              	movzbl	5(%rax,%rdi), %ebx
100004e39: 84 db                       	testb	%bl, %bl
100004e3b: 0f 48 de                    	cmovsl	%esi, %ebx
100004e3e: 88 5c 39 05                 	movb	%bl, 5(%rcx,%rdi)
100004e42: 0f b6 5c 38 06              	movzbl	6(%rax,%rdi), %ebx
100004e47: 84 db                       	testb	%bl, %bl
100004e49: 0f 48 de                    	cmovsl	%esi, %ebx
100004e4c: 88 5c 39 06                 	movb	%bl, 6(%rcx,%rdi)
100004e50: 0f b6 5c 38 07              	movzbl	7(%rax,%rdi), %ebx
100004e55: 84 db                       	testb	%bl, %bl
100004e57: 0f 48 de                    	cmovsl	%esi, %ebx
100004e5a: 88 5c 39 07                 	movb	%bl, 7(%rcx,%rdi)
100004e5e: 48 83 c7 08                 	addq	$8, %rdi
100004e62: 39 fa                       	cmpl	%edi, %edx
100004e64: 75 8a                       	jne	-118 <__ZN11LineNetwork7forwardEv+0xc00>
100004e66: 41 0f b6 40 24              	movzbl	36(%r8), %eax
100004e6b: 48 83 f0 01                 	xorq	$1, %rax
100004e6f: 41 88 40 24                 	movb	%al, 36(%r8)
100004e73: 31 c9                       	xorl	%ecx, %ecx
100004e75: 84 c0                       	testb	%al, %al
100004e77: 0f 94 c1                    	sete	%cl
100004e7a: 4d 8b 64 c8 28              	movq	40(%r8,%rcx,8), %r12
100004e7f: 49 8b 44 c0 28              	movq	40(%r8,%rax,8), %rax
100004e84: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100004e88: 31 c0                       	xorl	%eax, %eax
100004e8a: eb 1c                       	jmp	28 <__ZN11LineNetwork7forwardEv+0xcb8>
100004e8c: 0f 1f 40 00                 	nopl	(%rax)
100004e90: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100004e94: 48 ff c0                    	incq	%rax
100004e97: 4c 8b 65 c0                 	movq	-64(%rbp), %r12
100004e9b: 49 ff c4                    	incq	%r12
100004e9e: 48 83 f8 10                 	cmpq	$16, %rax
100004ea2: 0f 84 04 01 00 00           	je	260 <__ZN11LineNetwork7forwardEv+0xdbc>
100004ea8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100004eac: 48 8d 04 c0                 	leaq	(%rax,%rax,8), %rax
100004eb0: 48 8d 0d 89 27 00 00        	leaq	10121(%rip), %rcx
100004eb7: 48 8d 14 c1                 	leaq	(%rcx,%rax,8), %rdx
100004ebb: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100004ebf: 48 8d 54 c1 18              	leaq	24(%rcx,%rax,8), %rdx
100004ec4: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100004ec8: 48 8d 44 c1 30              	leaq	48(%rcx,%rax,8), %rax
100004ecd: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100004ed1: 4c 89 65 c0                 	movq	%r12, -64(%rbp)
100004ed5: 48 8b 5d c8                 	movq	-56(%rbp), %rbx
100004ed9: 31 c0                       	xorl	%eax, %eax
100004edb: eb 22                       	jmp	34 <__ZN11LineNetwork7forwardEv+0xd0f>
100004edd: 0f 1f 00                    	nopl	(%rax)
100004ee0: 48 8b 45 b8                 	movq	-72(%rbp), %rax
100004ee4: 48 ff c0                    	incq	%rax
100004ee7: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100004eeb: 48 81 c3 f0 07 00 00        	addq	$2032, %rbx
100004ef2: 49 81 c4 f0 03 00 00        	addq	$1008, %r12
100004ef9: 48 83 f8 3f                 	cmpq	$63, %rax
100004efd: 74 91                       	je	-111 <__ZN11LineNetwork7forwardEv+0xca0>
100004eff: 48 89 45 b8                 	movq	%rax, -72(%rbp)
100004f03: 45 31 ed                    	xorl	%r13d, %r13d
100004f06: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
100004f0a: eb 19                       	jmp	25 <__ZN11LineNetwork7forwardEv+0xd35>
100004f0c: 0f 1f 40 00                 	nopl	(%rax)
100004f10: 43 88 04 2c                 	movb	%al, (%r12,%r13)
100004f14: 48 83 c3 10                 	addq	$16, %rbx
100004f18: 49 83 c5 10                 	addq	$16, %r13
100004f1c: 49 81 fd f0 03 00 00        	cmpq	$1008, %r13
100004f23: 74 bb                       	je	-69 <__ZN11LineNetwork7forwardEv+0xcf0>
100004f25: 48 89 df                    	movq	%rbx, %rdi
100004f28: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100004f2c: c5 f8 77                    	vzeroupper
100004f2f: e8 2c 1b 00 00              	callq	6956 <__ZN11LineNetwork7forwardEv+0x2870>
100004f34: 41 89 c6                    	movl	%eax, %r14d
100004f37: 48 8d bb f8 03 00 00        	leaq	1016(%rbx), %rdi
100004f3e: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100004f42: e8 19 1b 00 00              	callq	6937 <__ZN11LineNetwork7forwardEv+0x2870>
100004f47: 41 89 c7                    	movl	%eax, %r15d
100004f4a: 45 01 f7                    	addl	%r14d, %r15d
100004f4d: 48 8d bb f0 07 00 00        	leaq	2032(%rbx), %rdi
100004f54: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100004f58: e8 03 1b 00 00              	callq	6915 <__ZN11LineNetwork7forwardEv+0x2870>
100004f5d: 44 01 f8                    	addl	%r15d, %eax
100004f60: 48 8b 4d d0                 	movq	-48(%rbp), %rcx
100004f64: 48 8d 15 55 2b 00 00        	leaq	11093(%rip), %rdx
100004f6b: 0f be 0c 11                 	movsbl	(%rcx,%rdx), %ecx
100004f6f: 01 c1                       	addl	%eax, %ecx
100004f71: 6b c1 39                    	imull	$57, %ecx, %eax
100004f74: 48 98                       	cltq
100004f76: 48 69 c8 09 04 02 81        	imulq	$-2130574327, %rax, %rcx
100004f7d: 48 c1 e9 20                 	shrq	$32, %rcx
100004f81: 01 c8                       	addl	%ecx, %eax
100004f83: 89 c1                       	movl	%eax, %ecx
100004f85: c1 e9 1f                    	shrl	$31, %ecx
100004f88: c1 f8 0d                    	sarl	$13, %eax
100004f8b: 01 c8                       	addl	%ecx, %eax
100004f8d: 3d 80 00 00 00              	cmpl	$128, %eax
100004f92: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0xda9>
100004f94: b8 7f 00 00 00              	movl	$127, %eax
100004f99: 83 f8 81                    	cmpl	$-127, %eax
100004f9c: 0f 8f 6e ff ff ff           	jg	-146 <__ZN11LineNetwork7forwardEv+0xd20>
100004fa2: b8 81 00 00 00              	movl	$129, %eax
100004fa7: e9 64 ff ff ff              	jmp	-156 <__ZN11LineNetwork7forwardEv+0xd20>
100004fac: 4c 8b 45 90                 	movq	-112(%rbp), %r8
100004fb0: 41 0f b6 40 24              	movzbl	36(%r8), %eax
100004fb5: 48 83 f0 01                 	xorq	$1, %rax
100004fb9: 41 88 40 24                 	movb	%al, 36(%r8)
100004fbd: 31 c9                       	xorl	%ecx, %ecx
100004fbf: 84 c0                       	testb	%al, %al
100004fc1: 0f 94 c1                    	sete	%cl
100004fc4: 49 8b 54 c8 28              	movq	40(%r8,%rcx,8), %rdx
100004fc9: 49 8b 74 c0 28              	movq	40(%r8,%rax,8), %rsi
100004fce: 48 8d 86 10 f8 00 00        	leaq	63504(%rsi), %rax
100004fd5: 48 39 c2                    	cmpq	%rax, %rdx
100004fd8: 73 1c                       	jae	28 <__ZN11LineNetwork7forwardEv+0xe06>
100004fda: 48 8d 82 10 f8 00 00        	leaq	63504(%rdx), %rax
100004fe1: bf 10 f8 00 00              	movl	$63504, %edi
100004fe6: 48 39 c6                    	cmpq	%rax, %rsi
100004fe9: 73 0b                       	jae	11 <__ZN11LineNetwork7forwardEv+0xe06>
100004feb: 48 89 f0                    	movq	%rsi, %rax
100004fee: 48 89 d1                    	movq	%rdx, %rcx
100004ff1: e9 b4 04 00 00              	jmp	1204 <__ZN11LineNetwork7forwardEv+0x12ba>
100004ff6: 48 8d 86 00 f8 00 00        	leaq	63488(%rsi), %rax
100004ffd: 48 8d 8a 00 f8 00 00        	leaq	63488(%rdx), %rcx
100005004: bf 20 00 00 00              	movl	$32, %edi
100005009: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005010: c5 7a 6f 74 be 80           	vmovdqu	-128(%rsi,%rdi,4), %xmm14
100005016: c5 7a 6f 7c be 90           	vmovdqu	-112(%rsi,%rdi,4), %xmm15
10000501c: c5 fa 6f 54 be a0           	vmovdqu	-96(%rsi,%rdi,4), %xmm2
100005022: c5 fa 6f 5c be b0           	vmovdqu	-80(%rsi,%rdi,4), %xmm3
100005028: c5 79 6f 1d 40 21 00 00     	vmovdqa	8512(%rip), %xmm11
100005030: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100005035: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
10000503a: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000503e: c5 79 6f 05 3a 21 00 00     	vmovdqa	8506(%rip), %xmm8
100005046: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
10000504b: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
100005050: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005054: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
10000505a: c5 fa 6f 64 be f0           	vmovdqu	-16(%rsi,%rdi,4), %xmm4
100005060: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100005065: c4 e3 fd 00 6c be e0 4e     	vpermq	$78, -32(%rsi,%rdi,4), %ymm5
10000506d: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005073: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100005078: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000507c: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100005082: c5 fa 6f 74 be d0           	vmovdqu	-48(%rsi,%rdi,4), %xmm6
100005088: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
10000508d: c4 63 fd 00 4c be c0 4e     	vpermq	$78, -64(%rsi,%rdi,4), %ymm9
100005095: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
10000509b: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
1000050a0: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
1000050a5: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000050ab: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
1000050b1: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
1000050b7: c5 79 6f 05 d1 20 00 00     	vmovdqa	8401(%rip), %xmm8
1000050bf: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
1000050c4: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
1000050c9: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
1000050cd: c5 79 6f 1d cb 20 00 00     	vmovdqa	8395(%rip), %xmm11
1000050d5: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
1000050da: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
1000050df: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
1000050e3: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
1000050e9: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
1000050ee: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
1000050f3: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000050f7: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000050fd: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
100005102: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
100005107: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
10000510b: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005111: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005117: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
10000511d: c5 79 6f 1d 8b 20 00 00     	vmovdqa	8331(%rip), %xmm11
100005125: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
10000512a: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
10000512f: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
100005133: c5 f9 6f 0d 85 20 00 00     	vmovdqa	8325(%rip), %xmm1
10000513b: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
10000513f: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100005144: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100005149: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
10000514d: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100005153: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005158: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
10000515d: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005161: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005167: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
10000516c: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100005171: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100005175: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000517b: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005181: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100005187: c5 f9 6f 0d 41 20 00 00     	vmovdqa	8257(%rip), %xmm1
10000518f: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100005194: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100005199: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000519d: c5 f9 6f 15 3b 20 00 00     	vmovdqa	8251(%rip), %xmm2
1000051a5: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
1000051a9: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
1000051ae: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
1000051b3: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000051b7: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
1000051bd: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
1000051c2: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
1000051c7: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000051cb: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000051d1: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
1000051d6: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
1000051db: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000051df: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
1000051e5: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
1000051eb: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
1000051f1: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
1000051f6: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
1000051fb: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
100005200: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
100005205: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
10000520a: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
10000520e: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005212: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005216: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
10000521a: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
10000521e: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100005222: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005226: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
10000522a: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100005230: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
100005236: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
10000523c: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005242: c5 fe 7f 4c ba c0           	vmovdqu	%ymm1, -64(%rdx,%rdi,4)
100005248: c5 fe 7f 44 ba e0           	vmovdqu	%ymm0, -32(%rdx,%rdi,4)
10000524e: c5 fe 7f 6c ba a0           	vmovdqu	%ymm5, -96(%rdx,%rdi,4)
100005254: c5 fe 7f 54 ba 80           	vmovdqu	%ymm2, -128(%rdx,%rdi,4)
10000525a: c5 7a 6f 24 be              	vmovdqu	(%rsi,%rdi,4), %xmm12
10000525f: c5 7a 6f 6c be 10           	vmovdqu	16(%rsi,%rdi,4), %xmm13
100005265: c5 7a 6f 74 be 20           	vmovdqu	32(%rsi,%rdi,4), %xmm14
10000526b: c5 fa 6f 5c be 30           	vmovdqu	48(%rsi,%rdi,4), %xmm3
100005271: c5 f9 6f 05 f7 1e 00 00     	vmovdqa	7927(%rip), %xmm0
100005279: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
10000527e: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
100005283: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100005287: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000528b: c5 f9 6f 05 ed 1e 00 00     	vmovdqa	7917(%rip), %xmm0
100005293: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100005298: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
10000529d: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
1000052a1: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
1000052a5: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
1000052ab: c5 fa 6f 64 be 70           	vmovdqu	112(%rsi,%rdi,4), %xmm4
1000052b1: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
1000052b6: c4 e3 fd 00 6c be 60 4e     	vpermq	$78, 96(%rsi,%rdi,4), %ymm5
1000052be: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
1000052c4: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
1000052c9: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
1000052cd: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
1000052d3: c5 fa 6f 74 be 50           	vmovdqu	80(%rsi,%rdi,4), %xmm6
1000052d9: c4 e3 fd 00 7c be 40 4e     	vpermq	$78, 64(%rsi,%rdi,4), %ymm7
1000052e1: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
1000052e6: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000052ec: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
1000052f1: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
1000052f5: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000052fb: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
100005301: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
100005307: c5 79 6f 3d 81 1e 00 00     	vmovdqa	7809(%rip), %xmm15
10000530f: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
100005314: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
100005319: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
10000531d: c5 f9 6f 05 7b 1e 00 00     	vmovdqa	7803(%rip), %xmm0
100005325: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000532a: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
10000532f: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005333: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
100005339: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
10000533e: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
100005343: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005347: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
10000534d: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005352: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005357: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
10000535b: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005361: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005367: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
10000536d: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005372: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100005377: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
10000537b: c5 f9 6f 05 3d 1e 00 00     	vmovdqa	7741(%rip), %xmm0
100005383: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100005388: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
10000538d: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005391: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
100005397: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
10000539c: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
1000053a1: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000053a5: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
1000053aa: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
1000053af: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
1000053b3: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000053b9: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000053bf: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000053c5: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
1000053cb: c5 79 6f 3d fd 1d 00 00     	vmovdqa	7677(%rip), %xmm15
1000053d3: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
1000053d8: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
1000053dd: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000053e1: c5 f9 6f 05 f7 1d 00 00     	vmovdqa	7671(%rip), %xmm0
1000053e9: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
1000053ee: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
1000053f3: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000053f7: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
1000053fd: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
100005402: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
100005407: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000540b: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
100005410: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
100005415: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005419: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
10000541f: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100005425: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
10000542b: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
100005431: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
100005436: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
10000543b: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
100005440: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100005445: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100005449: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
10000544d: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005451: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100005455: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100005459: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
10000545d: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005461: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100005465: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
10000546b: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100005471: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
100005477: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
10000547d: c5 fe 7f 4c ba 40           	vmovdqu	%ymm1, 64(%rdx,%rdi,4)
100005483: c5 fe 7f 44 ba 60           	vmovdqu	%ymm0, 96(%rdx,%rdi,4)
100005489: c5 fe 7f 5c ba 20           	vmovdqu	%ymm3, 32(%rdx,%rdi,4)
10000548f: c5 fe 7f 14 ba              	vmovdqu	%ymm2, (%rdx,%rdi,4)
100005494: 48 83 c7 40                 	addq	$64, %rdi
100005498: 48 81 ff 20 3e 00 00        	cmpq	$15904, %rdi
10000549f: 0f 85 6b fb ff ff           	jne	-1173 <__ZN11LineNetwork7forwardEv+0xe20>
1000054a5: bf 10 00 00 00              	movl	$16, %edi
1000054aa: 89 fa                       	movl	%edi, %edx
1000054ac: 31 f6                       	xorl	%esi, %esi
1000054ae: 31 ff                       	xorl	%edi, %edi
1000054b0: 0f b6 1c 38                 	movzbl	(%rax,%rdi), %ebx
1000054b4: 84 db                       	testb	%bl, %bl
1000054b6: 0f 48 de                    	cmovsl	%esi, %ebx
1000054b9: 88 1c 39                    	movb	%bl, (%rcx,%rdi)
1000054bc: 0f b6 5c 38 01              	movzbl	1(%rax,%rdi), %ebx
1000054c1: 84 db                       	testb	%bl, %bl
1000054c3: 0f 48 de                    	cmovsl	%esi, %ebx
1000054c6: 88 5c 39 01                 	movb	%bl, 1(%rcx,%rdi)
1000054ca: 0f b6 5c 38 02              	movzbl	2(%rax,%rdi), %ebx
1000054cf: 84 db                       	testb	%bl, %bl
1000054d1: 0f 48 de                    	cmovsl	%esi, %ebx
1000054d4: 88 5c 39 02                 	movb	%bl, 2(%rcx,%rdi)
1000054d8: 0f b6 5c 38 03              	movzbl	3(%rax,%rdi), %ebx
1000054dd: 84 db                       	testb	%bl, %bl
1000054df: 0f 48 de                    	cmovsl	%esi, %ebx
1000054e2: 88 5c 39 03                 	movb	%bl, 3(%rcx,%rdi)
1000054e6: 0f b6 5c 38 04              	movzbl	4(%rax,%rdi), %ebx
1000054eb: 84 db                       	testb	%bl, %bl
1000054ed: 0f 48 de                    	cmovsl	%esi, %ebx
1000054f0: 88 5c 39 04                 	movb	%bl, 4(%rcx,%rdi)
1000054f4: 0f b6 5c 38 05              	movzbl	5(%rax,%rdi), %ebx
1000054f9: 84 db                       	testb	%bl, %bl
1000054fb: 0f 48 de                    	cmovsl	%esi, %ebx
1000054fe: 88 5c 39 05                 	movb	%bl, 5(%rcx,%rdi)
100005502: 0f b6 5c 38 06              	movzbl	6(%rax,%rdi), %ebx
100005507: 84 db                       	testb	%bl, %bl
100005509: 0f 48 de                    	cmovsl	%esi, %ebx
10000550c: 88 5c 39 06                 	movb	%bl, 6(%rcx,%rdi)
100005510: 0f b6 5c 38 07              	movzbl	7(%rax,%rdi), %ebx
100005515: 84 db                       	testb	%bl, %bl
100005517: 0f 48 de                    	cmovsl	%esi, %ebx
10000551a: 88 5c 39 07                 	movb	%bl, 7(%rcx,%rdi)
10000551e: 48 83 c7 08                 	addq	$8, %rdi
100005522: 39 fa                       	cmpl	%edi, %edx
100005524: 75 8a                       	jne	-118 <__ZN11LineNetwork7forwardEv+0x12c0>
100005526: 41 0f b6 40 24              	movzbl	36(%r8), %eax
10000552b: 48 83 f0 01                 	xorq	$1, %rax
10000552f: 41 88 40 24                 	movb	%al, 36(%r8)
100005533: 31 c9                       	xorl	%ecx, %ecx
100005535: 84 c0                       	testb	%al, %al
100005537: 0f 94 c1                    	sete	%cl
10000553a: 4d 8b 64 c8 28              	movq	40(%r8,%rcx,8), %r12
10000553f: 49 8b 44 c0 28              	movq	40(%r8,%rax,8), %rax
100005544: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100005548: 31 c0                       	xorl	%eax, %eax
10000554a: eb 1c                       	jmp	28 <__ZN11LineNetwork7forwardEv+0x1378>
10000554c: 0f 1f 40 00                 	nopl	(%rax)
100005550: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100005554: 48 ff c0                    	incq	%rax
100005557: 4c 8b 65 c0                 	movq	-64(%rbp), %r12
10000555b: 49 ff c4                    	incq	%r12
10000555e: 48 83 f8 20                 	cmpq	$32, %rax
100005562: 0f 84 17 01 00 00           	je	279 <__ZN11LineNetwork7forwardEv+0x148f>
100005568: 48 89 45 d0                 	movq	%rax, -48(%rbp)
10000556c: 48 8d 04 c0                 	leaq	(%rax,%rax,8), %rax
100005570: 48 c1 e0 04                 	shlq	$4, %rax
100005574: 48 8d 0d 55 25 00 00        	leaq	9557(%rip), %rcx
10000557b: 48 8d 14 01                 	leaq	(%rcx,%rax), %rdx
10000557f: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005583: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005587: 48 83 c2 30                 	addq	$48, %rdx
10000558b: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
10000558f: 48 8d 44 08 60              	leaq	96(%rax,%rcx), %rax
100005594: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100005598: 4c 89 65 c0                 	movq	%r12, -64(%rbp)
10000559c: 48 8b 5d c8                 	movq	-56(%rbp), %rbx
1000055a0: 31 c0                       	xorl	%eax, %eax
1000055a2: eb 2b                       	jmp	43 <__ZN11LineNetwork7forwardEv+0x13df>
1000055a4: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000055ae: 66 90                       	nop
1000055b0: 48 8b 45 b8                 	movq	-72(%rbp), %rax
1000055b4: 48 ff c0                    	incq	%rax
1000055b7: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
1000055bb: 48 81 c3 e0 07 00 00        	addq	$2016, %rbx
1000055c2: 49 81 c4 e0 03 00 00        	addq	$992, %r12
1000055c9: 48 83 f8 1f                 	cmpq	$31, %rax
1000055cd: 74 81                       	je	-127 <__ZN11LineNetwork7forwardEv+0x1360>
1000055cf: 48 89 45 b8                 	movq	%rax, -72(%rbp)
1000055d3: 45 31 ed                    	xorl	%r13d, %r13d
1000055d6: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
1000055da: eb 19                       	jmp	25 <__ZN11LineNetwork7forwardEv+0x1405>
1000055dc: 0f 1f 40 00                 	nopl	(%rax)
1000055e0: 43 88 04 2c                 	movb	%al, (%r12,%r13)
1000055e4: 48 83 c3 20                 	addq	$32, %rbx
1000055e8: 49 83 c5 20                 	addq	$32, %r13
1000055ec: 49 81 fd e0 03 00 00        	cmpq	$992, %r13
1000055f3: 74 bb                       	je	-69 <__ZN11LineNetwork7forwardEv+0x13c0>
1000055f5: 48 89 df                    	movq	%rbx, %rdi
1000055f8: 48 8b 75 98                 	movq	-104(%rbp), %rsi
1000055fc: c5 f8 77                    	vzeroupper
1000055ff: e8 1c 15 00 00              	callq	5404 <__ZN11LineNetwork7forwardEv+0x2930>
100005604: 41 89 c6                    	movl	%eax, %r14d
100005607: 48 8d bb f0 03 00 00        	leaq	1008(%rbx), %rdi
10000560e: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005612: e8 09 15 00 00              	callq	5385 <__ZN11LineNetwork7forwardEv+0x2930>
100005617: 41 89 c7                    	movl	%eax, %r15d
10000561a: 45 01 f7                    	addl	%r14d, %r15d
10000561d: 48 8d bb e0 07 00 00        	leaq	2016(%rbx), %rdi
100005624: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100005628: e8 f3 14 00 00              	callq	5363 <__ZN11LineNetwork7forwardEv+0x2930>
10000562d: 44 01 f8                    	addl	%r15d, %eax
100005630: 48 8b 4d d0                 	movq	-48(%rbp), %rcx
100005634: 48 8d 15 95 36 00 00        	leaq	13973(%rip), %rdx
10000563b: 0f be 0c 11                 	movsbl	(%rcx,%rdx), %ecx
10000563f: 01 c1                       	addl	%eax, %ecx
100005641: c1 e1 04                    	shll	$4, %ecx
100005644: 8d 04 49                    	leal	(%rcx,%rcx,2), %eax
100005647: 48 98                       	cltq
100005649: 48 69 c8 09 04 02 81        	imulq	$-2130574327, %rax, %rcx
100005650: 48 c1 e9 20                 	shrq	$32, %rcx
100005654: 01 c8                       	addl	%ecx, %eax
100005656: 89 c1                       	movl	%eax, %ecx
100005658: c1 e9 1f                    	shrl	$31, %ecx
10000565b: c1 f8 0d                    	sarl	$13, %eax
10000565e: 01 c8                       	addl	%ecx, %eax
100005660: 3d 80 00 00 00              	cmpl	$128, %eax
100005665: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x147c>
100005667: b8 7f 00 00 00              	movl	$127, %eax
10000566c: 83 f8 81                    	cmpl	$-127, %eax
10000566f: 0f 8f 6b ff ff ff           	jg	-149 <__ZN11LineNetwork7forwardEv+0x13f0>
100005675: b8 81 00 00 00              	movl	$129, %eax
10000567a: e9 61 ff ff ff              	jmp	-159 <__ZN11LineNetwork7forwardEv+0x13f0>
10000567f: 4c 8b 45 90                 	movq	-112(%rbp), %r8
100005683: 41 0f b6 40 24              	movzbl	36(%r8), %eax
100005688: 48 83 f0 01                 	xorq	$1, %rax
10000568c: 41 88 40 24                 	movb	%al, 36(%r8)
100005690: 31 c9                       	xorl	%ecx, %ecx
100005692: 84 c0                       	testb	%al, %al
100005694: 0f 94 c1                    	sete	%cl
100005697: 49 8b 54 c8 28              	movq	40(%r8,%rcx,8), %rdx
10000569c: 49 8b 74 c0 28              	movq	40(%r8,%rax,8), %rsi
1000056a1: 48 8d 86 20 78 00 00        	leaq	30752(%rsi), %rax
1000056a8: 48 39 c2                    	cmpq	%rax, %rdx
1000056ab: 73 1c                       	jae	28 <__ZN11LineNetwork7forwardEv+0x14d9>
1000056ad: 48 8d 82 20 78 00 00        	leaq	30752(%rdx), %rax
1000056b4: bf 20 78 00 00              	movl	$30752, %edi
1000056b9: 48 39 c6                    	cmpq	%rax, %rsi
1000056bc: 73 0b                       	jae	11 <__ZN11LineNetwork7forwardEv+0x14d9>
1000056be: 48 89 f0                    	movq	%rsi, %rax
1000056c1: 48 89 d1                    	movq	%rdx, %rcx
1000056c4: e9 b1 04 00 00              	jmp	1201 <__ZN11LineNetwork7forwardEv+0x198a>
1000056c9: 48 8d 86 00 78 00 00        	leaq	30720(%rsi), %rax
1000056d0: 48 8d 8a 00 78 00 00        	leaq	30720(%rdx), %rcx
1000056d7: bf 20 00 00 00              	movl	$32, %edi
1000056dc: 0f 1f 40 00                 	nopl	(%rax)
1000056e0: c5 7a 6f 74 be 80           	vmovdqu	-128(%rsi,%rdi,4), %xmm14
1000056e6: c5 7a 6f 7c be 90           	vmovdqu	-112(%rsi,%rdi,4), %xmm15
1000056ec: c5 fa 6f 54 be a0           	vmovdqu	-96(%rsi,%rdi,4), %xmm2
1000056f2: c5 fa 6f 5c be b0           	vmovdqu	-80(%rsi,%rdi,4), %xmm3
1000056f8: c5 79 6f 1d 70 1a 00 00     	vmovdqa	6768(%rip), %xmm11
100005700: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100005705: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
10000570a: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000570e: c5 79 6f 05 6a 1a 00 00     	vmovdqa	6762(%rip), %xmm8
100005716: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
10000571b: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
100005720: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005724: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
10000572a: c5 fa 6f 64 be f0           	vmovdqu	-16(%rsi,%rdi,4), %xmm4
100005730: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100005735: c4 e3 fd 00 6c be e0 4e     	vpermq	$78, -32(%rsi,%rdi,4), %ymm5
10000573d: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005743: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100005748: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000574c: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100005752: c5 fa 6f 74 be d0           	vmovdqu	-48(%rsi,%rdi,4), %xmm6
100005758: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
10000575d: c4 63 fd 00 4c be c0 4e     	vpermq	$78, -64(%rsi,%rdi,4), %ymm9
100005765: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
10000576b: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
100005770: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100005775: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000577b: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100005781: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005787: c5 79 6f 05 01 1a 00 00     	vmovdqa	6657(%rip), %xmm8
10000578f: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100005794: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100005799: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
10000579d: c5 79 6f 1d fb 19 00 00     	vmovdqa	6651(%rip), %xmm11
1000057a5: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
1000057aa: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
1000057af: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
1000057b3: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
1000057b9: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
1000057be: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
1000057c3: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000057c7: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000057cd: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
1000057d2: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
1000057d7: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
1000057db: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000057e1: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000057e7: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
1000057ed: c5 79 6f 1d bb 19 00 00     	vmovdqa	6587(%rip), %xmm11
1000057f5: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
1000057fa: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
1000057ff: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
100005803: c5 f9 6f 0d b5 19 00 00     	vmovdqa	6581(%rip), %xmm1
10000580b: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
10000580f: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100005814: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100005819: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
10000581d: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100005823: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005828: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
10000582d: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005831: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005837: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
10000583c: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100005841: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100005845: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000584b: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005851: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100005857: c5 f9 6f 0d 71 19 00 00     	vmovdqa	6513(%rip), %xmm1
10000585f: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100005864: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100005869: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000586d: c5 f9 6f 15 6b 19 00 00     	vmovdqa	6507(%rip), %xmm2
100005875: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100005879: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
10000587e: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100005883: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005887: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
10000588d: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100005892: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100005897: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000589b: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000058a1: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
1000058a6: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
1000058ab: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000058af: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
1000058b5: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
1000058bb: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
1000058c1: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
1000058c6: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
1000058cb: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
1000058d0: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
1000058d5: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000058da: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
1000058de: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000058e2: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
1000058e6: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
1000058ea: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000058ee: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000058f2: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000058f6: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000058fa: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100005900: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
100005906: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
10000590c: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005912: c5 fe 7f 4c ba c0           	vmovdqu	%ymm1, -64(%rdx,%rdi,4)
100005918: c5 fe 7f 44 ba e0           	vmovdqu	%ymm0, -32(%rdx,%rdi,4)
10000591e: c5 fe 7f 6c ba a0           	vmovdqu	%ymm5, -96(%rdx,%rdi,4)
100005924: c5 fe 7f 54 ba 80           	vmovdqu	%ymm2, -128(%rdx,%rdi,4)
10000592a: c5 7a 6f 24 be              	vmovdqu	(%rsi,%rdi,4), %xmm12
10000592f: c5 7a 6f 6c be 10           	vmovdqu	16(%rsi,%rdi,4), %xmm13
100005935: c5 7a 6f 74 be 20           	vmovdqu	32(%rsi,%rdi,4), %xmm14
10000593b: c5 fa 6f 5c be 30           	vmovdqu	48(%rsi,%rdi,4), %xmm3
100005941: c5 f9 6f 05 27 18 00 00     	vmovdqa	6183(%rip), %xmm0
100005949: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
10000594e: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
100005953: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100005957: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000595b: c5 f9 6f 05 1d 18 00 00     	vmovdqa	6173(%rip), %xmm0
100005963: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100005968: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
10000596d: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
100005971: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005975: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
10000597b: c5 fa 6f 64 be 70           	vmovdqu	112(%rsi,%rdi,4), %xmm4
100005981: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100005986: c4 e3 fd 00 6c be 60 4e     	vpermq	$78, 96(%rsi,%rdi,4), %ymm5
10000598e: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005994: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
100005999: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000599d: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
1000059a3: c5 fa 6f 74 be 50           	vmovdqu	80(%rsi,%rdi,4), %xmm6
1000059a9: c4 e3 fd 00 7c be 40 4e     	vpermq	$78, 64(%rsi,%rdi,4), %ymm7
1000059b1: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
1000059b6: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000059bc: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
1000059c1: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
1000059c5: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000059cb: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
1000059d1: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
1000059d7: c5 79 6f 3d b1 17 00 00     	vmovdqa	6065(%rip), %xmm15
1000059df: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
1000059e4: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
1000059e9: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
1000059ed: c5 f9 6f 05 ab 17 00 00     	vmovdqa	6059(%rip), %xmm0
1000059f5: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000059fa: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000059ff: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005a03: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
100005a09: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
100005a0e: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
100005a13: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005a17: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005a1d: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005a22: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005a27: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100005a2b: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005a31: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005a37: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005a3d: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005a42: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100005a47: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100005a4b: c5 f9 6f 05 6d 17 00 00     	vmovdqa	5997(%rip), %xmm0
100005a53: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100005a58: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005a5d: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005a61: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
100005a67: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005a6c: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100005a71: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005a75: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005a7a: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005a7f: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100005a83: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005a89: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005a8f: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005a95: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
100005a9b: c5 79 6f 3d 2d 17 00 00     	vmovdqa	5933(%rip), %xmm15
100005aa3: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
100005aa8: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
100005aad: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005ab1: c5 f9 6f 05 27 17 00 00     	vmovdqa	5927(%rip), %xmm0
100005ab9: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
100005abe: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
100005ac3: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005ac7: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
100005acd: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
100005ad2: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
100005ad7: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005adb: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
100005ae0: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
100005ae5: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005ae9: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005aef: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100005af5: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100005afb: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
100005b01: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
100005b06: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
100005b0b: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
100005b10: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100005b15: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100005b19: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005b1d: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005b21: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100005b25: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100005b29: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100005b2d: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005b31: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100005b35: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100005b3b: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100005b41: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
100005b47: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005b4d: c5 fe 7f 4c ba 40           	vmovdqu	%ymm1, 64(%rdx,%rdi,4)
100005b53: c5 fe 7f 44 ba 60           	vmovdqu	%ymm0, 96(%rdx,%rdi,4)
100005b59: c5 fe 7f 5c ba 20           	vmovdqu	%ymm3, 32(%rdx,%rdi,4)
100005b5f: c5 fe 7f 14 ba              	vmovdqu	%ymm2, (%rdx,%rdi,4)
100005b64: 48 83 c7 40                 	addq	$64, %rdi
100005b68: 48 81 ff 20 1e 00 00        	cmpq	$7712, %rdi
100005b6f: 0f 85 6b fb ff ff           	jne	-1173 <__ZN11LineNetwork7forwardEv+0x14f0>
100005b75: bf 20 00 00 00              	movl	$32, %edi
100005b7a: 89 fa                       	movl	%edi, %edx
100005b7c: 31 f6                       	xorl	%esi, %esi
100005b7e: 31 ff                       	xorl	%edi, %edi
100005b80: 0f b6 1c 38                 	movzbl	(%rax,%rdi), %ebx
100005b84: 84 db                       	testb	%bl, %bl
100005b86: 0f 48 de                    	cmovsl	%esi, %ebx
100005b89: 88 1c 39                    	movb	%bl, (%rcx,%rdi)
100005b8c: 0f b6 5c 38 01              	movzbl	1(%rax,%rdi), %ebx
100005b91: 84 db                       	testb	%bl, %bl
100005b93: 0f 48 de                    	cmovsl	%esi, %ebx
100005b96: 88 5c 39 01                 	movb	%bl, 1(%rcx,%rdi)
100005b9a: 0f b6 5c 38 02              	movzbl	2(%rax,%rdi), %ebx
100005b9f: 84 db                       	testb	%bl, %bl
100005ba1: 0f 48 de                    	cmovsl	%esi, %ebx
100005ba4: 88 5c 39 02                 	movb	%bl, 2(%rcx,%rdi)
100005ba8: 0f b6 5c 38 03              	movzbl	3(%rax,%rdi), %ebx
100005bad: 84 db                       	testb	%bl, %bl
100005baf: 0f 48 de                    	cmovsl	%esi, %ebx
100005bb2: 88 5c 39 03                 	movb	%bl, 3(%rcx,%rdi)
100005bb6: 0f b6 5c 38 04              	movzbl	4(%rax,%rdi), %ebx
100005bbb: 84 db                       	testb	%bl, %bl
100005bbd: 0f 48 de                    	cmovsl	%esi, %ebx
100005bc0: 88 5c 39 04                 	movb	%bl, 4(%rcx,%rdi)
100005bc4: 0f b6 5c 38 05              	movzbl	5(%rax,%rdi), %ebx
100005bc9: 84 db                       	testb	%bl, %bl
100005bcb: 0f 48 de                    	cmovsl	%esi, %ebx
100005bce: 88 5c 39 05                 	movb	%bl, 5(%rcx,%rdi)
100005bd2: 0f b6 5c 38 06              	movzbl	6(%rax,%rdi), %ebx
100005bd7: 84 db                       	testb	%bl, %bl
100005bd9: 0f 48 de                    	cmovsl	%esi, %ebx
100005bdc: 88 5c 39 06                 	movb	%bl, 6(%rcx,%rdi)
100005be0: 0f b6 5c 38 07              	movzbl	7(%rax,%rdi), %ebx
100005be5: 84 db                       	testb	%bl, %bl
100005be7: 0f 48 de                    	cmovsl	%esi, %ebx
100005bea: 88 5c 39 07                 	movb	%bl, 7(%rcx,%rdi)
100005bee: 48 83 c7 08                 	addq	$8, %rdi
100005bf2: 39 fa                       	cmpl	%edi, %edx
100005bf4: 75 8a                       	jne	-118 <__ZN11LineNetwork7forwardEv+0x1990>
100005bf6: 41 0f b6 40 24              	movzbl	36(%r8), %eax
100005bfb: 48 83 f0 01                 	xorq	$1, %rax
100005bff: 41 88 40 24                 	movb	%al, 36(%r8)
100005c03: 31 c9                       	xorl	%ecx, %ecx
100005c05: 84 c0                       	testb	%al, %al
100005c07: 0f 94 c1                    	sete	%cl
100005c0a: 4d 8b 7c c8 28              	movq	40(%r8,%rcx,8), %r15
100005c0f: 4d 8b 64 c0 28              	movq	40(%r8,%rax,8), %r12
100005c14: 31 c0                       	xorl	%eax, %eax
100005c16: 4c 8d 35 d3 30 00 00        	leaq	12499(%rip), %r14
100005c1d: eb 19                       	jmp	25 <__ZN11LineNetwork7forwardEv+0x1a48>
100005c1f: 90                          	nop
100005c20: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100005c24: 48 ff c0                    	incq	%rax
100005c27: 49 83 c7 1f                 	addq	$31, %r15
100005c2b: 49 81 c4 e0 03 00 00        	addq	$992, %r12
100005c32: 48 83 f8 1f                 	cmpq	$31, %rax
100005c36: 74 79                       	je	121 <__ZN11LineNetwork7forwardEv+0x1ac1>
100005c38: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100005c3c: 49 c7 c5 e1 ff ff ff        	movq	$-31, %r13
100005c43: 4c 89 e3                    	movq	%r12, %rbx
100005c46: eb 16                       	jmp	22 <__ZN11LineNetwork7forwardEv+0x1a6e>
100005c48: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100005c50: 43 88 44 2f 1f              	movb	%al, 31(%r15,%r13)
100005c55: 48 83 c3 20                 	addq	$32, %rbx
100005c59: 49 ff c5                    	incq	%r13
100005c5c: 74 c2                       	je	-62 <__ZN11LineNetwork7forwardEv+0x1a30>
100005c5e: 48 89 df                    	movq	%rbx, %rdi
100005c61: 4c 89 f6                    	movq	%r14, %rsi
100005c64: c5 f8 77                    	vzeroupper
100005c67: e8 34 11 00 00              	callq	4404 <__ZN11LineNetwork7forwardEv+0x2bb0>
100005c6c: c1 e0 05                    	shll	$5, %eax
100005c6f: 89 c1                       	movl	%eax, %ecx
100005c71: 83 c1 20                    	addl	$32, %ecx
100005c74: 48 63 c9                    	movslq	%ecx, %rcx
100005c77: 48 69 c9 09 04 02 81        	imulq	$-2130574327, %rcx, %rcx
100005c7e: 48 c1 e9 20                 	shrq	$32, %rcx
100005c82: 8d 04 01                    	leal	(%rcx,%rax), %eax
100005c85: 83 c0 20                    	addl	$32, %eax
100005c88: 89 c1                       	movl	%eax, %ecx
100005c8a: c1 e9 1f                    	shrl	$31, %ecx
100005c8d: c1 f8 0d                    	sarl	$13, %eax
100005c90: 01 c8                       	addl	%ecx, %eax
100005c92: 3d 80 00 00 00              	cmpl	$128, %eax
100005c97: 7d 07                       	jge	7 <__ZN11LineNetwork7forwardEv+0x1ab0>
100005c99: 83 f8 81                    	cmpl	$-127, %eax
100005c9c: 7f b2                       	jg	-78 <__ZN11LineNetwork7forwardEv+0x1a60>
100005c9e: eb 0a                       	jmp	10 <__ZN11LineNetwork7forwardEv+0x1aba>
100005ca0: b8 7f 00 00 00              	movl	$127, %eax
100005ca5: 83 f8 81                    	cmpl	$-127, %eax
100005ca8: 7f a6                       	jg	-90 <__ZN11LineNetwork7forwardEv+0x1a60>
100005caa: b8 81 00 00 00              	movl	$129, %eax
100005caf: eb 9f                       	jmp	-97 <__ZN11LineNetwork7forwardEv+0x1a60>
100005cb1: 48 83 c4 48                 	addq	$72, %rsp
100005cb5: 5b                          	popq	%rbx
100005cb6: 41 5c                       	popq	%r12
100005cb8: 41 5d                       	popq	%r13
100005cba: 41 5e                       	popq	%r14
100005cbc: 41 5f                       	popq	%r15
100005cbe: 5d                          	popq	%rbp
100005cbf: c3                          	retq
100005cc0: 55                          	pushq	%rbp
100005cc1: 48 89 e5                    	movq	%rsp, %rbp
100005cc4: 41 57                       	pushq	%r15
100005cc6: 41 56                       	pushq	%r14
100005cc8: 41 55                       	pushq	%r13
100005cca: 41 54                       	pushq	%r12
100005ccc: 53                          	pushq	%rbx
100005ccd: 48 83 e4 e0                 	andq	$-32, %rsp
100005cd1: 48 81 ec e0 02 00 00        	subq	$736, %rsp
100005cd8: 48 89 4c 24 48              	movq	%rcx, 72(%rsp)
100005cdd: 48 89 54 24 40              	movq	%rdx, 64(%rsp)
100005ce2: 49 89 fd                    	movq	%rdi, %r13
100005ce5: c4 c1 79 6e c0              	vmovd	%r8d, %xmm0
100005cea: c4 e2 7d 58 c8              	vpbroadcastd	%xmm0, %ymm1
100005cef: 48 8d 86 01 04 00 00        	leaq	1025(%rsi), %rax
100005cf6: 48 89 44 24 38              	movq	%rax, 56(%rsp)
100005cfb: 48 8d 86 02 04 00 00        	leaq	1026(%rsi), %rax
100005d02: 48 89 44 24 30              	movq	%rax, 48(%rsp)
100005d07: 45 31 f6                    	xorl	%r14d, %r14d
100005d0a: 41 bf 7f 00 00 00           	movl	$127, %r15d
100005d10: 41 bc 81 00 00 00           	movl	$129, %r12d
100005d16: c5 fd 6f 15 02 16 00 00     	vmovdqa	5634(%rip), %ymm2
100005d1e: 48 89 7c 24 20              	movq	%rdi, 32(%rsp)
100005d23: 44 89 44 24 14              	movl	%r8d, 20(%rsp)
100005d28: 48 89 74 24 58              	movq	%rsi, 88(%rsp)
100005d2d: c5 fd 7f 8c 24 60 02 00 00  	vmovdqa	%ymm1, 608(%rsp)
100005d36: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005d40: 49 8d 86 f1 07 00 00        	leaq	2033(%r14), %rax
100005d47: 48 89 84 24 80 00 00 00     	movq	%rax, 128(%rsp)
100005d4f: 4b 8d 04 f6                 	leaq	(%r14,%r14,8), %rax
100005d53: 48 8b 54 24 40              	movq	64(%rsp), %rdx
100005d58: 48 8d 0c 02                 	leaq	(%rdx,%rax), %rcx
100005d5c: 48 83 c1 09                 	addq	$9, %rcx
100005d60: 48 89 4c 24 78              	movq	%rcx, 120(%rsp)
100005d65: 4c 8b 4c 24 48              	movq	72(%rsp), %r9
100005d6a: 4b 8d 4c 31 01              	leaq	1(%r9,%r14), %rcx
100005d6f: 48 89 4c 24 70              	movq	%rcx, 112(%rsp)
100005d74: 4c 8d 14 02                 	leaq	(%rdx,%rax), %r10
100005d78: 4f 8d 0c 31                 	leaq	(%r9,%r14), %r9
100005d7c: 48 8d 44 02 08              	leaq	8(%rdx,%rax), %rax
100005d81: 48 89 44 24 68              	movq	%rax, 104(%rsp)
100005d86: c4 c1 f9 6e c6              	vmovq	%r14, %xmm0
100005d8b: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005d90: 4c 8b 5c 24 30              	movq	48(%rsp), %r11
100005d95: 4c 89 6c 24 50              	movq	%r13, 80(%rsp)
100005d9a: 48 8b 44 24 38              	movq	56(%rsp), %rax
100005d9f: 31 c9                       	xorl	%ecx, %ecx
100005da1: 4c 89 74 24 60              	movq	%r14, 96(%rsp)
100005da6: 4c 89 54 24 08              	movq	%r10, 8(%rsp)
100005dab: 4c 89 0c 24                 	movq	%r9, (%rsp)
100005daf: c5 fd 7f 84 24 80 02 00 00  	vmovdqa	%ymm0, 640(%rsp)
100005db8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100005dc0: 48 89 44 24 28              	movq	%rax, 40(%rsp)
100005dc5: 4c 89 ac 24 88 00 00 00     	movq	%r13, 136(%rsp)
100005dcd: 4c 89 9c 24 90 00 00 00     	movq	%r11, 144(%rsp)
100005dd5: 48 69 d9 f8 07 00 00        	imulq	$2040, %rcx, %rbx
100005ddc: 49 8d 14 1e                 	leaq	(%r14,%rbx), %rdx
100005de0: 48 8b 44 24 20              	movq	32(%rsp), %rax
100005de5: 48 01 c2                    	addq	%rax, %rdx
100005de8: 48 03 9c 24 80 00 00 00     	addq	128(%rsp), %rbx
100005df0: 48 01 c3                    	addq	%rax, %rbx
100005df3: 48 89 4c 24 18              	movq	%rcx, 24(%rsp)
100005df8: 48 89 c8                    	movq	%rcx, %rax
100005dfb: 48 c1 e0 0a                 	shlq	$10, %rax
100005dff: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100005e03: 48 81 c1 ff 05 00 00        	addq	$1535, %rcx
100005e0a: 48 01 f0                    	addq	%rsi, %rax
100005e0d: 48 39 ca                    	cmpq	%rcx, %rdx
100005e10: 41 0f 92 c3                 	setb	%r11b
100005e14: 48 39 d8                    	cmpq	%rbx, %rax
100005e17: 0f 92 c1                    	setb	%cl
100005e1a: 48 3b 54 24 78              	cmpq	120(%rsp), %rdx
100005e1f: 0f 92 c0                    	setb	%al
100005e22: 48 39 5c 24 68              	cmpq	%rbx, 104(%rsp)
100005e27: 41 0f 92 c2                 	setb	%r10b
100005e2b: 48 3b 54 24 70              	cmpq	112(%rsp), %rdx
100005e30: 0f 92 c2                    	setb	%dl
100005e33: 49 39 d9                    	cmpq	%rbx, %r9
100005e36: 41 0f 92 c1                 	setb	%r9b
100005e3a: 41 84 cb                    	testb	%cl, %r11b
100005e3d: 0f 85 bd 0a 00 00           	jne	2749 <__ZN11LineNetwork7forwardEv+0x2710>
100005e43: 44 20 d0                    	andb	%r10b, %al
100005e46: 0f 85 b4 0a 00 00           	jne	2740 <__ZN11LineNetwork7forwardEv+0x2710>
100005e4c: b8 00 00 00 00              	movl	$0, %eax
100005e51: 44 20 ca                    	andb	%r9b, %dl
100005e54: 4c 8b 54 24 08              	movq	8(%rsp), %r10
100005e59: 4c 8b 0c 24                 	movq	(%rsp), %r9
100005e5d: 0f 85 a8 0a 00 00           	jne	2728 <__ZN11LineNetwork7forwardEv+0x271b>
100005e63: 48 8b 4c 24 18              	movq	24(%rsp), %rcx
100005e68: 48 89 c8                    	movq	%rcx, %rax
100005e6b: 48 c1 e0 08                 	shlq	$8, %rax
100005e6f: 48 29 c8                    	subq	%rcx, %rax
100005e72: c4 e1 f9 6e c0              	vmovq	%rax, %xmm0
100005e77: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005e7c: c5 fd 7f 84 24 a0 02 00 00  	vmovdqa	%ymm0, 672(%rsp)
100005e85: 31 db                       	xorl	%ebx, %ebx
100005e87: c5 fc 28 05 71 14 00 00     	vmovaps	5233(%rip), %ymm0
100005e8f: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100005e98: c5 fc 28 05 40 14 00 00     	vmovaps	5184(%rip), %ymm0
100005ea0: c5 fc 29 84 24 20 02 00 00  	vmovaps	%ymm0, 544(%rsp)
100005ea9: c5 fc 28 05 0f 14 00 00     	vmovaps	5135(%rip), %ymm0
100005eb1: c5 fc 29 84 24 00 02 00 00  	vmovaps	%ymm0, 512(%rsp)
100005eba: c5 fc 28 05 de 13 00 00     	vmovaps	5086(%rip), %ymm0
100005ec2: c5 fc 29 84 24 e0 01 00 00  	vmovaps	%ymm0, 480(%rsp)
100005ecb: c5 fc 28 05 ad 13 00 00     	vmovaps	5037(%rip), %ymm0
100005ed3: c5 fc 29 84 24 c0 01 00 00  	vmovaps	%ymm0, 448(%rsp)
100005edc: c5 fc 28 05 7c 13 00 00     	vmovaps	4988(%rip), %ymm0
100005ee4: c5 fc 29 84 24 a0 01 00 00  	vmovaps	%ymm0, 416(%rsp)
100005eed: c5 fc 28 05 4b 13 00 00     	vmovaps	4939(%rip), %ymm0
100005ef5: c5 fc 29 84 24 80 01 00 00  	vmovaps	%ymm0, 384(%rsp)
100005efe: c5 fc 28 05 1a 13 00 00     	vmovaps	4890(%rip), %ymm0
100005f06: c5 fc 29 84 24 60 01 00 00  	vmovaps	%ymm0, 352(%rsp)
100005f0f: 90                          	nop
100005f10: 48 8b 44 24 28              	movq	40(%rsp), %rax
100005f15: c5 fe 6f 84 58 1f fc ff ff  	vmovdqu	-993(%rax,%rbx,2), %ymm0
100005f1e: c4 e2 7d 00 c2              	vpshufb	%ymm2, %ymm0, %ymm0
100005f23: c5 fe 6f 8c 58 ff fb ff ff  	vmovdqu	-1025(%rax,%rbx,2), %ymm1
100005f2c: c5 fe 6f ac 58 00 fc ff ff  	vmovdqu	-1024(%rax,%rbx,2), %ymm5
100005f35: c5 7d 6f 15 03 14 00 00     	vmovdqa	5123(%rip), %ymm10
100005f3d: c4 c2 75 00 ca              	vpshufb	%ymm10, %ymm1, %ymm1
100005f42: c4 e3 75 02 c0 cc           	vpblendd	$204, %ymm0, %ymm1, %ymm0
100005f48: c4 e3 fd 00 c8 d8           	vpermq	$216, %ymm0, %ymm1
100005f4e: c4 e3 fd 00 c0 db           	vpermq	$219, %ymm0, %ymm0
100005f54: c5 fa 6f 94 58 0f fc ff ff  	vmovdqu	-1009(%rax,%rbx,2), %xmm2
100005f5d: c5 f9 6f 1d 8b 12 00 00     	vmovdqa	4747(%rip), %xmm3
100005f65: c4 e2 69 00 d3              	vpshufb	%xmm3, %xmm2, %xmm2
100005f6a: c5 79 6f e3                 	vmovdqa	%xmm3, %xmm12
100005f6e: c4 e2 7d 21 f2              	vpmovsxbd	%xmm2, %ymm6
100005f73: c4 e2 7d 21 f9              	vpmovsxbd	%xmm1, %ymm7
100005f78: c4 c2 79 78 12              	vpbroadcastb	(%r10), %xmm2
100005f7d: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
100005f82: c4 62 6d 40 de              	vpmulld	%ymm6, %ymm2, %ymm11
100005f87: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100005f8c: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
100005f92: c5 fe 6f b4 58 20 fc ff ff  	vmovdqu	-992(%rax,%rbx,2), %ymm6
100005f9b: c4 62 4d 00 05 7c 13 00 00  	vpshufb	4988(%rip), %ymm6, %ymm8
100005fa4: c4 42 55 00 ca              	vpshufb	%ymm10, %ymm5, %ymm9
100005fa9: c4 43 35 02 c0 cc           	vpblendd	$204, %ymm8, %ymm9, %ymm8
100005faf: c4 62 7d 21 c9              	vpmovsxbd	%xmm1, %ymm9
100005fb4: c4 43 fd 00 e8 d8           	vpermq	$216, %ymm8, %ymm13
100005fba: c5 fd 6f 0d 9e 13 00 00     	vmovdqa	5022(%rip), %ymm1
100005fc2: c4 e2 4d 00 c9              	vpshufb	%ymm1, %ymm6, %ymm1
100005fc7: c5 fd 6f 1d b1 13 00 00     	vmovdqa	5041(%rip), %ymm3
100005fcf: c4 e2 55 00 eb              	vpshufb	%ymm3, %ymm5, %ymm5
100005fd4: c4 e3 55 02 e9 cc           	vpblendd	$204, %ymm1, %ymm5, %ymm5
100005fda: c4 e2 6d 40 c0              	vpmulld	%ymm0, %ymm2, %ymm0
100005fdf: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100005fe8: c4 e3 fd 00 f5 d8           	vpermq	$216, %ymm5, %ymm6
100005fee: c4 42 7d 21 fd              	vpmovsxbd	%xmm13, %ymm15
100005ff3: c4 c3 fd 00 c8 db           	vpermq	$219, %ymm8, %ymm1
100005ff9: c4 62 7d 21 c1              	vpmovsxbd	%xmm1, %ymm8
100005ffe: c4 e2 6d 40 c7              	vpmulld	%ymm7, %ymm2, %ymm0
100006003: c5 fd 7f 84 24 a0 00 00 00  	vmovdqa	%ymm0, 160(%rsp)
10000600c: c4 43 7d 39 ed 01           	vextracti128	$1, %ymm13, %xmm13
100006012: c5 fa 6f a4 58 10 fc ff ff  	vmovdqu	-1008(%rax,%rbx,2), %xmm4
10000601b: c4 c2 59 00 dc              	vpshufb	%xmm12, %xmm4, %xmm3
100006020: c4 c2 79 78 7a 01           	vpbroadcastb	1(%r10), %xmm7
100006026: c4 42 7d 21 f5              	vpmovsxbd	%xmm13, %ymm14
10000602b: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
100006030: c4 62 7d 21 e7              	vpmovsxbd	%xmm7, %ymm12
100006035: c4 e2 1d 40 db              	vpmulld	%ymm3, %ymm12, %ymm3
10000603a: c4 c2 1d 40 f8              	vpmulld	%ymm8, %ymm12, %ymm7
10000603f: c4 c3 7d 39 f5 01           	vextracti128	$1, %ymm6, %xmm13
100006045: c4 e3 fd 00 ed db           	vpermq	$219, %ymm5, %ymm5
10000604b: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006050: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
100006055: c4 42 1d 40 c7              	vpmulld	%ymm15, %ymm12, %ymm8
10000605a: c5 f9 6f 05 9e 11 00 00     	vmovdqa	4510(%rip), %xmm0
100006062: c4 e2 59 00 e0              	vpshufb	%xmm0, %xmm4, %xmm4
100006067: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
10000606c: c4 c2 79 78 4a 02           	vpbroadcastb	2(%r10), %xmm1
100006072: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
100006077: c4 42 7d 21 fd              	vpmovsxbd	%xmm13, %ymm15
10000607c: c4 62 75 40 ee              	vpmulld	%ymm6, %ymm1, %ymm13
100006081: c4 e2 75 40 f5              	vpmulld	%ymm5, %ymm1, %ymm6
100006086: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
10000608b: c5 fd 7f 84 24 40 01 00 00  	vmovdqa	%ymm0, 320(%rsp)
100006094: c4 e2 75 40 e4              	vpmulld	%ymm4, %ymm1, %ymm4
100006099: c5 fe 6f ac 58 ff fd ff ff  	vmovdqu	-513(%rax,%rbx,2), %ymm5
1000060a2: c5 7e 6f 8c 58 1f fe ff ff  	vmovdqu	-481(%rax,%rbx,2), %ymm9
1000060ab: c4 c2 1d 40 c6              	vpmulld	%ymm14, %ymm12, %ymm0
1000060b0: c5 fd 7f 84 24 00 01 00 00  	vmovdqa	%ymm0, 256(%rsp)
1000060b9: c4 62 35 00 0d 5e 12 00 00  	vpshufb	4702(%rip), %ymm9, %ymm9
1000060c2: c4 c2 55 00 ea              	vpshufb	%ymm10, %ymm5, %ymm5
1000060c7: c4 c3 55 02 e9 cc           	vpblendd	$204, %ymm9, %ymm5, %ymm5
1000060cd: c4 63 fd 00 cd d8           	vpermq	$216, %ymm5, %ymm9
1000060d3: c4 c2 75 40 c7              	vpmulld	%ymm15, %ymm1, %ymm0
1000060d8: c5 fd 7f 84 24 20 01 00 00  	vmovdqa	%ymm0, 288(%rsp)
1000060e1: c4 c2 7d 21 c9              	vpmovsxbd	%xmm9, %ymm1
1000060e6: c4 e3 fd 00 ed db           	vpermq	$219, %ymm5, %ymm5
1000060ec: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
1000060f1: c4 63 7d 39 ca 01           	vextracti128	$1, %ymm9, %xmm2
1000060f7: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000060fc: c5 25 fe fb                 	vpaddd	%ymm3, %ymm11, %ymm15
100006100: c5 fa 6f 9c 58 0f fe ff ff  	vmovdqu	-497(%rax,%rbx,2), %xmm3
100006109: c4 c2 79 78 42 03           	vpbroadcastb	3(%r10), %xmm0
10000610f: c4 62 7d 21 c8              	vpmovsxbd	%xmm0, %ymm9
100006114: c4 e2 35 40 c2              	vpmulld	%ymm2, %ymm9, %ymm0
100006119: c5 fd 7f 84 24 e0 00 00 00  	vmovdqa	%ymm0, 224(%rsp)
100006122: c5 79 6f 25 c6 10 00 00     	vmovdqa	4294(%rip), %xmm12
10000612a: c4 c2 61 00 d4              	vpshufb	%xmm12, %xmm3, %xmm2
10000612f: c4 e2 35 40 dd              	vpmulld	%ymm5, %ymm9, %ymm3
100006134: c4 e2 35 40 c9              	vpmulld	%ymm1, %ymm9, %ymm1
100006139: c5 45 fe b4 24 c0 00 00 00  	vpaddd	192(%rsp), %ymm7, %ymm14
100006142: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
100006147: c4 e2 35 40 d2              	vpmulld	%ymm2, %ymm9, %ymm2
10000614c: c5 5d fe ca                 	vpaddd	%ymm2, %ymm4, %ymm9
100006150: c5 fe 6f 94 58 00 fe ff ff  	vmovdqu	-512(%rax,%rbx,2), %ymm2
100006159: c5 3d fe 84 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm8, %ymm8
100006162: c5 fe 6f a4 58 20 fe ff ff  	vmovdqu	-480(%rax,%rbx,2), %ymm4
10000616b: c4 e2 5d 00 2d ac 11 00 00  	vpshufb	4524(%rip), %ymm4, %ymm5
100006174: c4 c2 6d 00 fa              	vpshufb	%ymm10, %ymm2, %ymm7
100006179: c4 e3 45 02 ed cc           	vpblendd	$204, %ymm5, %ymm7, %ymm5
10000617f: c4 63 fd 00 dd d8           	vpermq	$216, %ymm5, %ymm11
100006185: c5 cd fe fb                 	vpaddd	%ymm3, %ymm6, %ymm7
100006189: c4 e2 5d 00 1d ce 11 00 00  	vpshufb	4558(%rip), %ymm4, %ymm3
100006192: c4 e2 6d 00 15 e5 11 00 00  	vpshufb	4581(%rip), %ymm2, %ymm2
10000619b: c4 e3 6d 02 d3 cc           	vpblendd	$204, %ymm3, %ymm2, %ymm2
1000061a1: c4 63 7d 39 db 01           	vextracti128	$1, %ymm11, %xmm3
1000061a7: c4 e3 fd 00 e5 db           	vpermq	$219, %ymm5, %ymm4
1000061ad: c5 95 fe f1                 	vpaddd	%ymm1, %ymm13, %ymm6
1000061b1: c4 e2 7d 21 cc              	vpmovsxbd	%xmm4, %ymm1
1000061b6: c4 c2 7d 21 e3              	vpmovsxbd	%xmm11, %ymm4
1000061bb: c5 fa 6f 84 58 10 fe ff ff  	vmovdqu	-496(%rax,%rbx,2), %xmm0
1000061c4: c4 c2 79 78 6a 04           	vpbroadcastb	4(%r10), %xmm5
1000061ca: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000061cf: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
1000061d4: c4 e2 55 40 e4              	vpmulld	%ymm4, %ymm5, %ymm4
1000061d9: c4 62 55 40 d9              	vpmulld	%ymm1, %ymm5, %ymm11
1000061de: c4 c2 79 00 cc              	vpshufb	%xmm12, %xmm0, %xmm1
1000061e3: c4 62 7d 21 e9              	vpmovsxbd	%xmm1, %ymm13
1000061e8: c4 e2 55 40 cb              	vpmulld	%ymm3, %ymm5, %ymm1
1000061ed: c5 fd 7f 8c 24 a0 00 00 00  	vmovdqa	%ymm1, 160(%rsp)
1000061f6: c4 c2 55 40 dd              	vpmulld	%ymm13, %ymm5, %ymm3
1000061fb: c4 e3 fd 00 ea db           	vpermq	$219, %ymm2, %ymm5
100006201: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006206: c4 c2 79 78 4a 05           	vpbroadcastb	5(%r10), %xmm1
10000620c: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
100006211: c4 62 75 40 ed              	vpmulld	%ymm5, %ymm1, %ymm13
100006216: c4 41 05 fe f9              	vpaddd	%ymm9, %ymm15, %ymm15
10000621b: c4 41 25 fe cd              	vpaddd	%ymm13, %ymm11, %ymm9
100006220: c4 e3 fd 00 d2 d8           	vpermq	$216, %ymm2, %ymm2
100006226: c4 62 7d 21 da              	vpmovsxbd	%xmm2, %ymm11
10000622b: c4 42 75 40 db              	vpmulld	%ymm11, %ymm1, %ymm11
100006230: c4 c1 5d fe e3              	vpaddd	%ymm11, %ymm4, %ymm4
100006235: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000623b: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
100006240: c4 e2 79 00 05 b7 0f 00 00  	vpshufb	4023(%rip), %xmm0, %xmm0
100006249: c4 62 75 40 ea              	vpmulld	%ymm2, %ymm1, %ymm13
10000624e: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006253: c4 e2 75 40 c0              	vpmulld	%ymm0, %ymm1, %ymm0
100006258: c5 e5 fe c0                 	vpaddd	%ymm0, %ymm3, %ymm0
10000625c: c5 fe 6f 4c 58 ff           	vmovdqu	-1(%rax,%rbx,2), %ymm1
100006262: c5 8d fe d7                 	vpaddd	%ymm7, %ymm14, %ymm2
100006266: c5 fe 6f 5c 58 1f           	vmovdqu	31(%rax,%rbx,2), %ymm3
10000626c: c4 e2 65 00 1d ab 10 00 00  	vpshufb	4267(%rip), %ymm3, %ymm3
100006275: c4 c2 75 00 ca              	vpshufb	%ymm10, %ymm1, %ymm1
10000627a: c4 e3 75 02 cb cc           	vpblendd	$204, %ymm3, %ymm1, %ymm1
100006280: c4 e3 fd 00 d9 d8           	vpermq	$216, %ymm1, %ymm3
100006286: c5 bd fe f6                 	vpaddd	%ymm6, %ymm8, %ymm6
10000628a: c4 e2 7d 21 fb              	vpmovsxbd	%xmm3, %ymm7
10000628f: c4 c2 79 78 6a 06           	vpbroadcastb	6(%r10), %xmm5
100006295: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
10000629a: c4 e2 55 40 ff              	vpmulld	%ymm7, %ymm5, %ymm7
10000629f: c5 dd fe e7                 	vpaddd	%ymm7, %ymm4, %ymm4
1000062a3: c4 e3 7d 39 db 01           	vextracti128	$1, %ymm3, %xmm3
1000062a9: c4 e3 fd 00 c9 db           	vpermq	$219, %ymm1, %ymm1
1000062af: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000062b4: c4 e2 55 40 c9              	vpmulld	%ymm1, %ymm5, %ymm1
1000062b9: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000062be: c5 b5 fe f9                 	vpaddd	%ymm1, %ymm9, %ymm7
1000062c2: c5 fa 6f 4c 58 0f           	vmovdqu	15(%rax,%rbx,2), %xmm1
1000062c8: c4 c2 71 00 cc              	vpshufb	%xmm12, %xmm1, %xmm1
1000062cd: c4 62 7d 21 c1              	vpmovsxbd	%xmm1, %ymm8
1000062d2: c5 4d fe e4                 	vpaddd	%ymm4, %ymm6, %ymm12
1000062d6: c4 62 55 40 cb              	vpmulld	%ymm3, %ymm5, %ymm9
1000062db: c4 c2 55 40 d8              	vpmulld	%ymm8, %ymm5, %ymm3
1000062e0: c5 fd fe c3                 	vpaddd	%ymm3, %ymm0, %ymm0
1000062e4: c5 6d fe c7                 	vpaddd	%ymm7, %ymm2, %ymm8
1000062e8: c5 05 fe f0                 	vpaddd	%ymm0, %ymm15, %ymm14
1000062ec: c5 fe 6f 14 58              	vmovdqu	(%rax,%rbx,2), %ymm2
1000062f1: c5 fe 6f 5c 58 20           	vmovdqu	32(%rax,%rbx,2), %ymm3
1000062f7: c4 e2 65 00 25 20 10 00 00  	vpshufb	4128(%rip), %ymm3, %ymm4
100006300: c4 c2 6d 00 ea              	vpshufb	%ymm10, %ymm2, %ymm5
100006305: c5 fd 6f 84 24 00 01 00 00  	vmovdqa	256(%rsp), %ymm0
10000630e: c5 7d fe bc 24 40 01 00 00  	vpaddd	320(%rsp), %ymm0, %ymm15
100006317: c4 e3 55 02 e4 cc           	vpblendd	$204, %ymm4, %ymm5, %ymm4
10000631d: c4 e3 fd 00 ec d8           	vpermq	$216, %ymm4, %ymm5
100006323: c4 e2 65 00 1d 34 10 00 00  	vpshufb	4148(%rip), %ymm3, %ymm3
10000632c: c4 e2 6d 00 15 4b 10 00 00  	vpshufb	4171(%rip), %ymm2, %ymm2
100006335: c4 e3 6d 02 d3 cc           	vpblendd	$204, %ymm3, %ymm2, %ymm2
10000633b: c5 fd 6f 84 24 e0 00 00 00  	vmovdqa	224(%rsp), %ymm0
100006344: c5 fd fe 9c 24 20 01 00 00  	vpaddd	288(%rsp), %ymm0, %ymm3
10000634d: c4 e3 fd 00 fa d8           	vpermq	$216, %ymm2, %ymm7
100006353: c4 62 7d 21 dd              	vpmovsxbd	%xmm5, %ymm11
100006358: c4 e3 fd 00 e4 db           	vpermq	$219, %ymm4, %ymm4
10000635e: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100006363: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100006369: c5 15 fe 94 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm13, %ymm10
100006372: c5 fa 6f 44 58 10           	vmovdqu	16(%rax,%rbx,2), %xmm0
100006378: c4 e2 79 00 35 6f 0e 00 00  	vpshufb	3695(%rip), %xmm0, %xmm6
100006381: c4 c2 79 78 4a 07           	vpbroadcastb	7(%r10), %xmm1
100006387: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
10000638c: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006391: c4 e2 75 40 e4              	vpmulld	%ymm4, %ymm1, %ymm4
100006396: c4 42 75 40 db              	vpmulld	%ymm11, %ymm1, %ymm11
10000639b: c4 e2 75 40 ed              	vpmulld	%ymm5, %ymm1, %ymm5
1000063a0: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
1000063a5: c4 e2 75 40 ce              	vpmulld	%ymm6, %ymm1, %ymm1
1000063aa: c4 c2 79 78 72 08           	vpbroadcastb	8(%r10), %xmm6
1000063b0: c5 85 fe db                 	vpaddd	%ymm3, %ymm15, %ymm3
1000063b4: c4 62 7d 21 ef              	vpmovsxbd	%xmm7, %ymm13
1000063b9: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
1000063be: c4 42 4d 40 ed              	vpmulld	%ymm13, %ymm6, %ymm13
1000063c3: c4 41 25 fe dd              	vpaddd	%ymm13, %ymm11, %ymm11
1000063c8: c4 41 2d fe c9              	vpaddd	%ymm9, %ymm10, %ymm9
1000063cd: c4 e3 fd 00 d2 db           	vpermq	$219, %ymm2, %ymm2
1000063d3: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000063d8: c4 e2 4d 40 d2              	vpmulld	%ymm2, %ymm6, %ymm2
1000063dd: c5 dd fe d2                 	vpaddd	%ymm2, %ymm4, %ymm2
1000063e1: c4 c1 65 fe d9              	vpaddd	%ymm9, %ymm3, %ymm3
1000063e6: c4 e3 7d 39 fc 01           	vextracti128	$1, %ymm7, %xmm4
1000063ec: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000063f1: c4 e2 4d 40 e4              	vpmulld	%ymm4, %ymm6, %ymm4
1000063f6: c5 d5 fe e4                 	vpaddd	%ymm4, %ymm5, %ymm4
1000063fa: c4 e2 79 00 05 fd 0d 00 00  	vpshufb	3581(%rip), %xmm0, %xmm0
100006403: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006408: c4 e2 4d 40 c0              	vpmulld	%ymm0, %ymm6, %ymm0
10000640d: c5 f5 fe c0                 	vpaddd	%ymm0, %ymm1, %ymm0
100006411: c4 c2 79 78 09              	vpbroadcastb	(%r9), %xmm1
100006416: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
10000641b: c5 a5 fe e9                 	vpaddd	%ymm1, %ymm11, %ymm5
10000641f: c5 9d fe ed                 	vpaddd	%ymm5, %ymm12, %ymm5
100006423: c5 ed fe d1                 	vpaddd	%ymm1, %ymm2, %ymm2
100006427: c5 bd fe d2                 	vpaddd	%ymm2, %ymm8, %ymm2
10000642b: c5 dd fe e1                 	vpaddd	%ymm1, %ymm4, %ymm4
10000642f: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
100006433: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006437: c5 fd 6f a4 24 60 02 00 00  	vmovdqa	608(%rsp), %ymm4
100006440: c4 e2 6d 40 d4              	vpmulld	%ymm4, %ymm2, %ymm2
100006445: c4 e2 65 40 cc              	vpmulld	%ymm4, %ymm3, %ymm1
10000644a: c5 8d fe c0                 	vpaddd	%ymm0, %ymm14, %ymm0
10000644e: c4 e2 55 40 dc              	vpmulld	%ymm4, %ymm5, %ymm3
100006453: c4 e2 7d 40 c4              	vpmulld	%ymm4, %ymm0, %ymm0
100006458: c5 fd 70 e3 f5              	vpshufd	$245, %ymm3, %ymm4
10000645d: c4 e2 7d 58 2d ba 28 00 00  	vpbroadcastd	10426(%rip), %ymm5
100006466: c4 e2 5d 28 e5              	vpmuldq	%ymm5, %ymm4, %ymm4
10000646b: c4 e2 65 28 f5              	vpmuldq	%ymm5, %ymm3, %ymm6
100006470: c5 fd 70 f6 f5              	vpshufd	$245, %ymm6, %ymm6
100006475: c4 e3 4d 02 e4 aa           	vpblendd	$170, %ymm4, %ymm6, %ymm4
10000647b: c5 dd fe db                 	vpaddd	%ymm3, %ymm4, %ymm3
10000647f: c5 dd 72 d3 1f              	vpsrld	$31, %ymm3, %ymm4
100006484: c5 e5 72 e3 0d              	vpsrad	$13, %ymm3, %ymm3
100006489: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
10000648d: c5 fd 70 e2 f5              	vpshufd	$245, %ymm2, %ymm4
100006492: c4 e2 6d 28 f5              	vpmuldq	%ymm5, %ymm2, %ymm6
100006497: c4 e2 5d 28 e5              	vpmuldq	%ymm5, %ymm4, %ymm4
10000649c: c5 fd 70 f6 f5              	vpshufd	$245, %ymm6, %ymm6
1000064a1: c4 e3 4d 02 e4 aa           	vpblendd	$170, %ymm4, %ymm6, %ymm4
1000064a7: c5 dd fe d2                 	vpaddd	%ymm2, %ymm4, %ymm2
1000064ab: c5 dd 72 d2 1f              	vpsrld	$31, %ymm2, %ymm4
1000064b0: c5 ed 72 e2 0d              	vpsrad	$13, %ymm2, %ymm2
1000064b5: c5 ed fe d4                 	vpaddd	%ymm4, %ymm2, %ymm2
1000064b9: c5 fd 70 e1 f5              	vpshufd	$245, %ymm1, %ymm4
1000064be: c4 e2 5d 28 e5              	vpmuldq	%ymm5, %ymm4, %ymm4
1000064c3: c4 e2 75 28 f5              	vpmuldq	%ymm5, %ymm1, %ymm6
1000064c8: c5 fd 70 f6 f5              	vpshufd	$245, %ymm6, %ymm6
1000064cd: c4 e3 4d 02 e4 aa           	vpblendd	$170, %ymm4, %ymm6, %ymm4
1000064d3: c5 fd 70 f0 f5              	vpshufd	$245, %ymm0, %ymm6
1000064d8: c4 e2 4d 28 f5              	vpmuldq	%ymm5, %ymm6, %ymm6
1000064dd: c4 e2 7d 28 ed              	vpmuldq	%ymm5, %ymm0, %ymm5
1000064e2: c5 fd 70 ed f5              	vpshufd	$245, %ymm5, %ymm5
1000064e7: c4 e3 55 02 ee aa           	vpblendd	$170, %ymm6, %ymm5, %ymm5
1000064ed: c5 d5 fe c0                 	vpaddd	%ymm0, %ymm5, %ymm0
1000064f1: c5 dd fe c9                 	vpaddd	%ymm1, %ymm4, %ymm1
1000064f5: c5 dd 72 d0 1f              	vpsrld	$31, %ymm0, %ymm4
1000064fa: c5 fd 72 e0 0d              	vpsrad	$13, %ymm0, %ymm0
1000064ff: c5 fd fe c4                 	vpaddd	%ymm4, %ymm0, %ymm0
100006503: c5 dd 72 d1 1f              	vpsrld	$31, %ymm1, %ymm4
100006508: c5 f5 72 e1 0d              	vpsrad	$13, %ymm1, %ymm1
10000650d: c5 f5 fe cc                 	vpaddd	%ymm4, %ymm1, %ymm1
100006511: c4 e2 7d 58 25 0a 28 00 00  	vpbroadcastd	10250(%rip), %ymm4
10000651a: c4 e2 75 39 cc              	vpminsd	%ymm4, %ymm1, %ymm1
10000651f: c4 e2 6d 39 d4              	vpminsd	%ymm4, %ymm2, %ymm2
100006524: c4 e2 65 39 dc              	vpminsd	%ymm4, %ymm3, %ymm3
100006529: c4 e2 7d 39 c4              	vpminsd	%ymm4, %ymm0, %ymm0
10000652e: c4 e2 7d 58 25 f1 27 00 00  	vpbroadcastd	10225(%rip), %ymm4
100006537: c4 e2 65 3d dc              	vpmaxsd	%ymm4, %ymm3, %ymm3
10000653c: c4 e2 6d 3d d4              	vpmaxsd	%ymm4, %ymm2, %ymm2
100006541: c4 e2 75 3d ec              	vpmaxsd	%ymm4, %ymm1, %ymm5
100006546: c4 e2 7d 3d c4              	vpmaxsd	%ymm4, %ymm0, %ymm0
10000654b: c5 e5 6b c8                 	vpackssdw	%ymm0, %ymm3, %ymm1
10000654f: c5 d5 6b c2                 	vpackssdw	%ymm2, %ymm5, %ymm0
100006553: c5 fd 6f ac 24 a0 02 00 00  	vmovdqa	672(%rsp), %ymm5
10000655c: c5 fd 6f b4 24 40 02 00 00  	vmovdqa	576(%rsp), %ymm6
100006565: c5 cd d4 d5                 	vpaddq	%ymm5, %ymm6, %ymm2
100006569: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000656e: c5 fd 6f a4 24 80 02 00 00  	vmovdqa	640(%rsp), %ymm4
100006577: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
10000657b: c4 c1 f9 7e d4              	vmovq	%xmm2, %r12
100006580: c4 c3 f9 16 d7 01           	vpextrq	$1, %xmm2, %r15
100006586: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000658c: c4 e1 f9 7e d0              	vmovq	%xmm2, %rax
100006591: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
100006597: c5 fd 6f bc 24 20 02 00 00  	vmovdqa	544(%rsp), %ymm7
1000065a0: c5 c5 d4 d5                 	vpaddq	%ymm5, %ymm7, %ymm2
1000065a4: c4 e3 fd 00 c9 d8           	vpermq	$216, %ymm1, %ymm1
1000065aa: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000065af: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000065b3: c4 c1 f9 7e d6              	vmovq	%xmm2, %r14
1000065b8: c4 c3 f9 16 d0 01           	vpextrq	$1, %xmm2, %r8
1000065be: c4 e3 fd 00 d8 d8           	vpermq	$216, %ymm0, %ymm3
1000065c4: c4 e3 7d 39 d0 01           	vextracti128	$1, %ymm2, %xmm0
1000065ca: c4 c1 f9 7e c2              	vmovq	%xmm0, %r10
1000065cf: c4 e3 f9 16 c7 01           	vpextrq	$1, %xmm0, %rdi
1000065d5: c5 7d 6f 94 24 a0 01 00 00  	vmovdqa	416(%rsp), %ymm10
1000065de: c5 ad d4 c5                 	vpaddq	%ymm5, %ymm10, %ymm0
1000065e2: c5 f5 63 cb                 	vpacksswb	%ymm3, %ymm1, %ymm1
1000065e6: c5 fd 6f 9c 24 00 02 00 00  	vmovdqa	512(%rsp), %ymm3
1000065ef: c5 e5 d4 d5                 	vpaddq	%ymm5, %ymm3, %ymm2
1000065f3: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000065f8: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000065fc: c4 e1 f9 7e d2              	vmovq	%xmm2, %rdx
100006601: c4 e3 f9 16 94 24 20 01 00 00 01    	vpextrq	$1, %xmm2, 288(%rsp)
10000660c: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006612: c5 f9 d6 94 24 40 01 00 00  	vmovq	%xmm2, 320(%rsp)
10000661b: c4 c3 f9 16 d3 01           	vpextrq	$1, %xmm2, %r11
100006621: c5 7d 6f 84 24 e0 01 00 00  	vmovdqa	480(%rsp), %ymm8
10000662a: c5 bd d4 d5                 	vpaddq	%ymm5, %ymm8, %ymm2
10000662e: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006633: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006637: c4 c3 f9 16 d5 01           	vpextrq	$1, %xmm2, %r13
10000663d: 48 8b 4c 24 20              	movq	32(%rsp), %rcx
100006642: c4 a3 79 14 0c 21 00        	vpextrb	$0, %xmm1, (%rcx,%r12)
100006649: c4 e1 f9 7e d6              	vmovq	%xmm2, %rsi
10000664e: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006654: c4 a3 79 14 0c 39 01        	vpextrb	$1, %xmm1, (%rcx,%r15)
10000665b: c4 c1 f9 7e d4              	vmovq	%xmm2, %r12
100006660: c4 e3 79 14 0c 01 02        	vpextrb	$2, %xmm1, (%rcx,%rax)
100006667: c4 e3 f9 16 94 24 00 01 00 00 01    	vpextrq	$1, %xmm2, 256(%rsp)
100006672: c5 7d 6f 8c 24 c0 01 00 00  	vmovdqa	448(%rsp), %ymm9
10000667b: c5 b5 d4 d5                 	vpaddq	%ymm5, %ymm9, %ymm2
10000667f: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006684: c5 fd 73 f0 03              	vpsllq	$3, %ymm0, %ymm0
100006689: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
10000668d: c4 a3 79 14 0c 09 03        	vpextrb	$3, %xmm1, (%rcx,%r9)
100006694: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006698: c4 e3 f9 16 94 24 e0 00 00 00 01    	vpextrq	$1, %xmm2, 224(%rsp)
1000066a3: c4 a3 79 14 0c 31 04        	vpextrb	$4, %xmm1, (%rcx,%r14)
1000066aa: c4 c1 f9 7e d6              	vmovq	%xmm2, %r14
1000066af: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000066b5: c4 a3 79 14 0c 01 05        	vpextrb	$5, %xmm1, (%rcx,%r8)
1000066bc: c5 f9 d6 94 24 a0 00 00 00  	vmovq	%xmm2, 160(%rsp)
1000066c5: c4 a3 79 14 0c 11 06        	vpextrb	$6, %xmm1, (%rcx,%r10)
1000066cc: c4 e3 79 14 0c 39 07        	vpextrb	$7, %xmm1, (%rcx,%rdi)
1000066d3: c4 e3 f9 16 94 24 c0 00 00 00 01    	vpextrq	$1, %xmm2, 192(%rsp)
1000066de: c4 e3 7d 39 ca 01           	vextracti128	$1, %ymm1, %xmm2
1000066e4: c4 e3 79 14 14 11 00        	vpextrb	$0, %xmm2, (%rcx,%rdx)
1000066eb: c5 f9 d6 84 24 98 00 00 00  	vmovq	%xmm0, 152(%rsp)
1000066f4: c4 c3 f9 16 c2 01           	vpextrq	$1, %xmm0, %r10
1000066fa: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100006702: c4 e3 79 14 14 01 01        	vpextrb	$1, %xmm2, (%rcx,%rax)
100006709: c4 e3 7d 39 c0 01           	vextracti128	$1, %ymm0, %xmm0
10000670f: c4 c3 f9 16 c0 01           	vpextrq	$1, %xmm0, %r8
100006715: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
10000671d: c4 e3 79 14 14 01 02        	vpextrb	$2, %xmm2, (%rcx,%rax)
100006724: c4 c1 f9 7e c7              	vmovq	%xmm0, %r15
100006729: c5 7d 6f 9c 24 80 01 00 00  	vmovdqa	384(%rsp), %ymm11
100006732: c5 a5 d4 c5                 	vpaddq	%ymm5, %ymm11, %ymm0
100006736: c5 fd 73 f0 03              	vpsllq	$3, %ymm0, %ymm0
10000673b: c4 a3 79 14 14 19 03        	vpextrb	$3, %xmm2, (%rcx,%r11)
100006742: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
100006746: c4 c3 f9 16 c1 01           	vpextrq	$1, %xmm0, %r9
10000674c: c4 e3 79 14 14 31 04        	vpextrb	$4, %xmm2, (%rcx,%rsi)
100006753: c4 e1 f9 7e c7              	vmovq	%xmm0, %rdi
100006758: c4 e3 7d 39 c0 01           	vextracti128	$1, %ymm0, %xmm0
10000675e: c4 a3 79 14 14 29 05        	vpextrb	$5, %xmm2, (%rcx,%r13)
100006765: c4 e1 f9 7e c0              	vmovq	%xmm0, %rax
10000676a: c4 a3 79 14 14 21 06        	vpextrb	$6, %xmm2, (%rcx,%r12)
100006771: c4 c3 f9 16 c4 01           	vpextrq	$1, %xmm0, %r12
100006777: c5 7d 6f a4 24 60 01 00 00  	vmovdqa	352(%rsp), %ymm12
100006780: c5 9d d4 c5                 	vpaddq	%ymm5, %ymm12, %ymm0
100006784: c5 fd 73 f0 03              	vpsllq	$3, %ymm0, %ymm0
100006789: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
10000678d: 48 8b b4 24 00 01 00 00     	movq	256(%rsp), %rsi
100006795: c4 e3 79 14 14 31 07        	vpextrb	$7, %xmm2, (%rcx,%rsi)
10000679c: c4 c1 f9 7e c5              	vmovq	%xmm0, %r13
1000067a1: c4 a3 79 14 0c 31 08        	vpextrb	$8, %xmm1, (%rcx,%r14)
1000067a8: c4 c3 f9 16 c6 01           	vpextrq	$1, %xmm0, %r14
1000067ae: c4 e3 7d 39 c0 01           	vextracti128	$1, %ymm0, %xmm0
1000067b4: 48 8b b4 24 e0 00 00 00     	movq	224(%rsp), %rsi
1000067bc: c4 e3 79 14 0c 31 09        	vpextrb	$9, %xmm1, (%rcx,%rsi)
1000067c3: 48 8b b4 24 a0 00 00 00     	movq	160(%rsp), %rsi
1000067cb: c4 e3 79 14 0c 31 0a        	vpextrb	$10, %xmm1, (%rcx,%rsi)
1000067d2: c4 e1 f9 7e c6              	vmovq	%xmm0, %rsi
1000067d7: c4 c3 f9 16 c3 01           	vpextrq	$1, %xmm0, %r11
1000067dd: 48 8b 94 24 c0 00 00 00     	movq	192(%rsp), %rdx
1000067e5: c4 e3 79 14 0c 11 0b        	vpextrb	$11, %xmm1, (%rcx,%rdx)
1000067ec: 48 8b 94 24 98 00 00 00     	movq	152(%rsp), %rdx
1000067f4: c4 e3 79 14 0c 11 0c        	vpextrb	$12, %xmm1, (%rcx,%rdx)
1000067fb: c4 a3 79 14 0c 11 0d        	vpextrb	$13, %xmm1, (%rcx,%r10)
100006802: 4c 8b 54 24 08              	movq	8(%rsp), %r10
100006807: c4 a3 79 14 0c 39 0e        	vpextrb	$14, %xmm1, (%rcx,%r15)
10000680e: c4 a3 79 14 0c 01 0f        	vpextrb	$15, %xmm1, (%rcx,%r8)
100006815: c4 e3 79 14 14 39 08        	vpextrb	$8, %xmm2, (%rcx,%rdi)
10000681c: c4 a3 79 14 14 09 09        	vpextrb	$9, %xmm2, (%rcx,%r9)
100006823: 4c 8b 0c 24                 	movq	(%rsp), %r9
100006827: c4 e3 79 14 14 01 0a        	vpextrb	$10, %xmm2, (%rcx,%rax)
10000682e: c4 a3 79 14 14 21 0b        	vpextrb	$11, %xmm2, (%rcx,%r12)
100006835: c4 a3 79 14 14 29 0c        	vpextrb	$12, %xmm2, (%rcx,%r13)
10000683c: c4 a3 79 14 14 31 0d        	vpextrb	$13, %xmm2, (%rcx,%r14)
100006843: c4 e3 79 14 14 31 0e        	vpextrb	$14, %xmm2, (%rcx,%rsi)
10000684a: c4 a3 79 14 14 19 0f        	vpextrb	$15, %xmm2, (%rcx,%r11)
100006851: c5 fd 6f 15 c7 0a 00 00     	vmovdqa	2759(%rip), %ymm2
100006859: c4 e2 7d 59 05 ce 24 00 00  	vpbroadcastq	9422(%rip), %ymm0
100006862: c5 cd d4 f0                 	vpaddq	%ymm0, %ymm6, %ymm6
100006866: c5 fd 7f b4 24 40 02 00 00  	vmovdqa	%ymm6, 576(%rsp)
10000686f: c5 c5 d4 f8                 	vpaddq	%ymm0, %ymm7, %ymm7
100006873: c5 fd 7f bc 24 20 02 00 00  	vmovdqa	%ymm7, 544(%rsp)
10000687c: c5 e5 d4 d8                 	vpaddq	%ymm0, %ymm3, %ymm3
100006880: c5 fd 7f 9c 24 00 02 00 00  	vmovdqa	%ymm3, 512(%rsp)
100006889: c5 3d d4 c0                 	vpaddq	%ymm0, %ymm8, %ymm8
10000688d: c5 7d 7f 84 24 e0 01 00 00  	vmovdqa	%ymm8, 480(%rsp)
100006896: c5 35 d4 c8                 	vpaddq	%ymm0, %ymm9, %ymm9
10000689a: c5 7d 7f 8c 24 c0 01 00 00  	vmovdqa	%ymm9, 448(%rsp)
1000068a3: c5 2d d4 d0                 	vpaddq	%ymm0, %ymm10, %ymm10
1000068a7: c5 7d 7f 94 24 a0 01 00 00  	vmovdqa	%ymm10, 416(%rsp)
1000068b0: c5 25 d4 d8                 	vpaddq	%ymm0, %ymm11, %ymm11
1000068b4: c5 7d 7f 9c 24 80 01 00 00  	vmovdqa	%ymm11, 384(%rsp)
1000068bd: c5 1d d4 e0                 	vpaddq	%ymm0, %ymm12, %ymm12
1000068c1: c5 7d 7f a4 24 60 01 00 00  	vmovdqa	%ymm12, 352(%rsp)
1000068ca: 48 83 c3 20                 	addq	$32, %rbx
1000068ce: 48 81 fb e0 00 00 00        	cmpq	$224, %rbx
1000068d5: 0f 85 35 f6 ff ff           	jne	-2507 <__ZN11LineNetwork7forwardEv+0x1d20>
1000068db: b8 e0 00 00 00              	movl	$224, %eax
1000068e0: 44 8b 44 24 14              	movl	20(%rsp), %r8d
1000068e5: 48 8b 74 24 58              	movq	88(%rsp), %rsi
1000068ea: 4c 8b 74 24 60              	movq	96(%rsp), %r14
1000068ef: 41 bf 7f 00 00 00           	movl	$127, %r15d
1000068f5: 41 bc 81 00 00 00           	movl	$129, %r12d
1000068fb: eb 0e                       	jmp	14 <__ZN11LineNetwork7forwardEv+0x271b>
1000068fd: 0f 1f 00                    	nopl	(%rax)
100006900: 31 c0                       	xorl	%eax, %eax
100006902: 4c 8b 54 24 08              	movq	8(%rsp), %r10
100006907: 4c 8b 0c 24                 	movq	(%rsp), %r9
10000690b: 4c 8b 9c 24 90 00 00 00     	movq	144(%rsp), %r11
100006913: 4c 8b ac 24 88 00 00 00     	movq	136(%rsp), %r13
10000691b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100006920: 41 0f be 8c 43 fe fb ff ff  	movsbl	-1026(%r11,%rax,2), %ecx
100006929: 41 0f be 12                 	movsbl	(%r10), %edx
10000692d: 0f af d1                    	imull	%ecx, %edx
100006930: 41 0f be 8c 43 ff fb ff ff  	movsbl	-1025(%r11,%rax,2), %ecx
100006939: 41 0f be 5a 01              	movsbl	1(%r10), %ebx
10000693e: 0f af d9                    	imull	%ecx, %ebx
100006941: 01 d3                       	addl	%edx, %ebx
100006943: 41 0f be 8c 43 00 fc ff ff  	movsbl	-1024(%r11,%rax,2), %ecx
10000694c: 41 0f be 52 02              	movsbl	2(%r10), %edx
100006951: 0f af d1                    	imull	%ecx, %edx
100006954: 01 da                       	addl	%ebx, %edx
100006956: 41 0f be 8c 43 fe fd ff ff  	movsbl	-514(%r11,%rax,2), %ecx
10000695f: 41 0f be 5a 03              	movsbl	3(%r10), %ebx
100006964: 0f af d9                    	imull	%ecx, %ebx
100006967: 01 d3                       	addl	%edx, %ebx
100006969: 41 0f be 8c 43 ff fd ff ff  	movsbl	-513(%r11,%rax,2), %ecx
100006972: 41 0f be 52 04              	movsbl	4(%r10), %edx
100006977: 0f af d1                    	imull	%ecx, %edx
10000697a: 01 da                       	addl	%ebx, %edx
10000697c: 41 0f be 8c 43 00 fe ff ff  	movsbl	-512(%r11,%rax,2), %ecx
100006985: 41 0f be 5a 05              	movsbl	5(%r10), %ebx
10000698a: 0f af d9                    	imull	%ecx, %ebx
10000698d: 01 d3                       	addl	%edx, %ebx
10000698f: 41 0f be 4c 43 fe           	movsbl	-2(%r11,%rax,2), %ecx
100006995: 41 0f be 52 06              	movsbl	6(%r10), %edx
10000699a: 0f af d1                    	imull	%ecx, %edx
10000699d: 01 da                       	addl	%ebx, %edx
10000699f: 41 0f be 4c 43 ff           	movsbl	-1(%r11,%rax,2), %ecx
1000069a5: 41 0f be 5a 07              	movsbl	7(%r10), %ebx
1000069aa: 0f af d9                    	imull	%ecx, %ebx
1000069ad: 01 d3                       	addl	%edx, %ebx
1000069af: 41 0f be 0c 43              	movsbl	(%r11,%rax,2), %ecx
1000069b4: 41 0f be 52 08              	movsbl	8(%r10), %edx
1000069b9: 0f af d1                    	imull	%ecx, %edx
1000069bc: 01 da                       	addl	%ebx, %edx
1000069be: 41 0f be 09                 	movsbl	(%r9), %ecx
1000069c2: 01 d1                       	addl	%edx, %ecx
1000069c4: 41 0f af c8                 	imull	%r8d, %ecx
1000069c8: 48 63 c9                    	movslq	%ecx, %rcx
1000069cb: 48 69 d1 09 04 02 81        	imulq	$-2130574327, %rcx, %rdx
1000069d2: 48 c1 ea 20                 	shrq	$32, %rdx
1000069d6: 01 d1                       	addl	%edx, %ecx
1000069d8: 89 ca                       	movl	%ecx, %edx
1000069da: c1 ea 1f                    	shrl	$31, %edx
1000069dd: c1 f9 0d                    	sarl	$13, %ecx
1000069e0: 01 d1                       	addl	%edx, %ecx
1000069e2: 81 f9 80 00 00 00           	cmpl	$128, %ecx
1000069e8: 41 0f 4d cf                 	cmovgel	%r15d, %ecx
1000069ec: 83 f9 81                    	cmpl	$-127, %ecx
1000069ef: 41 0f 4e cc                 	cmovlel	%r12d, %ecx
1000069f3: 41 88 4c c5 00              	movb	%cl, (%r13,%rax,8)
1000069f8: 48 ff c0                    	incq	%rax
1000069fb: 48 3d ff 00 00 00           	cmpq	$255, %rax
100006a01: 0f 85 19 ff ff ff           	jne	-231 <__ZN11LineNetwork7forwardEv+0x2730>
100006a07: 48 8b 4c 24 18              	movq	24(%rsp), %rcx
100006a0c: 48 ff c1                    	incq	%rcx
100006a0f: 48 8b 44 24 28              	movq	40(%rsp), %rax
100006a14: 48 05 00 04 00 00           	addq	$1024, %rax
100006a1a: 49 81 c5 f8 07 00 00        	addq	$2040, %r13
100006a21: 49 81 c3 00 04 00 00        	addq	$1024, %r11
100006a28: 48 81 f9 ff 00 00 00        	cmpq	$255, %rcx
100006a2f: 0f 85 8b f3 ff ff           	jne	-3189 <__ZN11LineNetwork7forwardEv+0x1bd0>
100006a35: 49 ff c6                    	incq	%r14
100006a38: 4c 8b 6c 24 50              	movq	80(%rsp), %r13
100006a3d: 49 ff c5                    	incq	%r13
100006a40: 49 83 fe 08                 	cmpq	$8, %r14
100006a44: 0f 85 f6 f2 ff ff           	jne	-3338 <__ZN11LineNetwork7forwardEv+0x1b50>
100006a4a: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100006a4e: 5b                          	popq	%rbx
100006a4f: 41 5c                       	popq	%r12
100006a51: 41 5d                       	popq	%r13
100006a53: 41 5e                       	popq	%r14
100006a55: 41 5f                       	popq	%r15
100006a57: 5d                          	popq	%rbp
100006a58: c5 f8 77                    	vzeroupper
100006a5b: c3                          	retq
100006a5c: 0f 1f 40 00                 	nopl	(%rax)
100006a60: 55                          	pushq	%rbp
100006a61: 48 89 e5                    	movq	%rsp, %rbp
100006a64: c4 e2 7d 21 06              	vpmovsxbd	(%rsi), %ymm0
100006a69: c4 e2 7d 21 4e 08           	vpmovsxbd	8(%rsi), %ymm1
100006a6f: c4 e2 7d 21 17              	vpmovsxbd	(%rdi), %ymm2
100006a74: c4 e2 7d 40 c2              	vpmulld	%ymm2, %ymm0, %ymm0
100006a79: c4 e2 7d 21 57 08           	vpmovsxbd	8(%rdi), %ymm2
100006a7f: c4 e2 75 40 ca              	vpmulld	%ymm2, %ymm1, %ymm1
100006a84: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006a88: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006a8e: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006a92: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006a97: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006a9b: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006aa0: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006aa4: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006aa8: 0f be 4f 10                 	movsbl	16(%rdi), %ecx
100006aac: 0f be 56 10                 	movsbl	16(%rsi), %edx
100006ab0: 0f af d1                    	imull	%ecx, %edx
100006ab3: 01 c2                       	addl	%eax, %edx
100006ab5: 0f be 47 11                 	movsbl	17(%rdi), %eax
100006ab9: 0f be 4e 11                 	movsbl	17(%rsi), %ecx
100006abd: 0f af c8                    	imull	%eax, %ecx
100006ac0: 01 d1                       	addl	%edx, %ecx
100006ac2: 0f be 47 12                 	movsbl	18(%rdi), %eax
100006ac6: 0f be 56 12                 	movsbl	18(%rsi), %edx
100006aca: 0f af d0                    	imull	%eax, %edx
100006acd: 01 ca                       	addl	%ecx, %edx
100006acf: 0f be 47 13                 	movsbl	19(%rdi), %eax
100006ad3: 0f be 4e 13                 	movsbl	19(%rsi), %ecx
100006ad7: 0f af c8                    	imull	%eax, %ecx
100006ada: 01 d1                       	addl	%edx, %ecx
100006adc: 0f be 47 14                 	movsbl	20(%rdi), %eax
100006ae0: 0f be 56 14                 	movsbl	20(%rsi), %edx
100006ae4: 0f af d0                    	imull	%eax, %edx
100006ae7: 01 ca                       	addl	%ecx, %edx
100006ae9: 0f be 47 15                 	movsbl	21(%rdi), %eax
100006aed: 0f be 4e 15                 	movsbl	21(%rsi), %ecx
100006af1: 0f af c8                    	imull	%eax, %ecx
100006af4: 01 d1                       	addl	%edx, %ecx
100006af6: 0f be 47 16                 	movsbl	22(%rdi), %eax
100006afa: 0f be 56 16                 	movsbl	22(%rsi), %edx
100006afe: 0f af d0                    	imull	%eax, %edx
100006b01: 01 ca                       	addl	%ecx, %edx
100006b03: 0f be 4f 17                 	movsbl	23(%rdi), %ecx
100006b07: 0f be 46 17                 	movsbl	23(%rsi), %eax
100006b0b: 0f af c1                    	imull	%ecx, %eax
100006b0e: 01 d0                       	addl	%edx, %eax
100006b10: 5d                          	popq	%rbp
100006b11: c5 f8 77                    	vzeroupper
100006b14: c3                          	retq
100006b15: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100006b1f: 90                          	nop
100006b20: 55                          	pushq	%rbp
100006b21: 48 89 e5                    	movq	%rsp, %rbp
100006b24: 0f be 06                    	movsbl	(%rsi), %eax
100006b27: 0f be 0f                    	movsbl	(%rdi), %ecx
100006b2a: 0f af c8                    	imull	%eax, %ecx
100006b2d: 0f be 46 01                 	movsbl	1(%rsi), %eax
100006b31: 0f be 57 01                 	movsbl	1(%rdi), %edx
100006b35: 0f af d0                    	imull	%eax, %edx
100006b38: 01 ca                       	addl	%ecx, %edx
100006b3a: 0f be 46 02                 	movsbl	2(%rsi), %eax
100006b3e: 0f be 4f 02                 	movsbl	2(%rdi), %ecx
100006b42: 0f af c8                    	imull	%eax, %ecx
100006b45: 01 d1                       	addl	%edx, %ecx
100006b47: 0f be 46 03                 	movsbl	3(%rsi), %eax
100006b4b: 0f be 57 03                 	movsbl	3(%rdi), %edx
100006b4f: 0f af d0                    	imull	%eax, %edx
100006b52: 01 ca                       	addl	%ecx, %edx
100006b54: 0f be 46 04                 	movsbl	4(%rsi), %eax
100006b58: 0f be 4f 04                 	movsbl	4(%rdi), %ecx
100006b5c: 0f af c8                    	imull	%eax, %ecx
100006b5f: 01 d1                       	addl	%edx, %ecx
100006b61: 0f be 46 05                 	movsbl	5(%rsi), %eax
100006b65: 0f be 57 05                 	movsbl	5(%rdi), %edx
100006b69: 0f af d0                    	imull	%eax, %edx
100006b6c: 01 ca                       	addl	%ecx, %edx
100006b6e: 0f be 46 06                 	movsbl	6(%rsi), %eax
100006b72: 0f be 4f 06                 	movsbl	6(%rdi), %ecx
100006b76: 0f af c8                    	imull	%eax, %ecx
100006b79: 01 d1                       	addl	%edx, %ecx
100006b7b: 0f be 46 07                 	movsbl	7(%rsi), %eax
100006b7f: 0f be 57 07                 	movsbl	7(%rdi), %edx
100006b83: 0f af d0                    	imull	%eax, %edx
100006b86: 01 ca                       	addl	%ecx, %edx
100006b88: 0f be 46 08                 	movsbl	8(%rsi), %eax
100006b8c: 0f be 4f 08                 	movsbl	8(%rdi), %ecx
100006b90: 0f af c8                    	imull	%eax, %ecx
100006b93: 01 d1                       	addl	%edx, %ecx
100006b95: 0f be 46 09                 	movsbl	9(%rsi), %eax
100006b99: 0f be 57 09                 	movsbl	9(%rdi), %edx
100006b9d: 0f af d0                    	imull	%eax, %edx
100006ba0: 01 ca                       	addl	%ecx, %edx
100006ba2: 0f be 46 0a                 	movsbl	10(%rsi), %eax
100006ba6: 0f be 4f 0a                 	movsbl	10(%rdi), %ecx
100006baa: 0f af c8                    	imull	%eax, %ecx
100006bad: 01 d1                       	addl	%edx, %ecx
100006baf: 0f be 46 0b                 	movsbl	11(%rsi), %eax
100006bb3: 0f be 57 0b                 	movsbl	11(%rdi), %edx
100006bb7: 0f af d0                    	imull	%eax, %edx
100006bba: 01 ca                       	addl	%ecx, %edx
100006bbc: 0f be 46 0c                 	movsbl	12(%rsi), %eax
100006bc0: 0f be 4f 0c                 	movsbl	12(%rdi), %ecx
100006bc4: 0f af c8                    	imull	%eax, %ecx
100006bc7: 01 d1                       	addl	%edx, %ecx
100006bc9: 0f be 46 0d                 	movsbl	13(%rsi), %eax
100006bcd: 0f be 57 0d                 	movsbl	13(%rdi), %edx
100006bd1: 0f af d0                    	imull	%eax, %edx
100006bd4: 01 ca                       	addl	%ecx, %edx
100006bd6: 0f be 46 0e                 	movsbl	14(%rsi), %eax
100006bda: 0f be 4f 0e                 	movsbl	14(%rdi), %ecx
100006bde: 0f af c8                    	imull	%eax, %ecx
100006be1: 01 d1                       	addl	%edx, %ecx
100006be3: 0f be 46 0f                 	movsbl	15(%rsi), %eax
100006be7: 0f be 57 0f                 	movsbl	15(%rdi), %edx
100006beb: 0f af d0                    	imull	%eax, %edx
100006bee: 01 ca                       	addl	%ecx, %edx
100006bf0: 0f be 46 10                 	movsbl	16(%rsi), %eax
100006bf4: 0f be 4f 10                 	movsbl	16(%rdi), %ecx
100006bf8: 0f af c8                    	imull	%eax, %ecx
100006bfb: 01 d1                       	addl	%edx, %ecx
100006bfd: 0f be 46 11                 	movsbl	17(%rsi), %eax
100006c01: 0f be 57 11                 	movsbl	17(%rdi), %edx
100006c05: 0f af d0                    	imull	%eax, %edx
100006c08: 01 ca                       	addl	%ecx, %edx
100006c0a: 0f be 46 12                 	movsbl	18(%rsi), %eax
100006c0e: 0f be 4f 12                 	movsbl	18(%rdi), %ecx
100006c12: 0f af c8                    	imull	%eax, %ecx
100006c15: 01 d1                       	addl	%edx, %ecx
100006c17: 0f be 46 13                 	movsbl	19(%rsi), %eax
100006c1b: 0f be 57 13                 	movsbl	19(%rdi), %edx
100006c1f: 0f af d0                    	imull	%eax, %edx
100006c22: 01 ca                       	addl	%ecx, %edx
100006c24: 0f be 46 14                 	movsbl	20(%rsi), %eax
100006c28: 0f be 4f 14                 	movsbl	20(%rdi), %ecx
100006c2c: 0f af c8                    	imull	%eax, %ecx
100006c2f: 01 d1                       	addl	%edx, %ecx
100006c31: 0f be 46 15                 	movsbl	21(%rsi), %eax
100006c35: 0f be 57 15                 	movsbl	21(%rdi), %edx
100006c39: 0f af d0                    	imull	%eax, %edx
100006c3c: 01 ca                       	addl	%ecx, %edx
100006c3e: 0f be 46 16                 	movsbl	22(%rsi), %eax
100006c42: 0f be 4f 16                 	movsbl	22(%rdi), %ecx
100006c46: 0f af c8                    	imull	%eax, %ecx
100006c49: 01 d1                       	addl	%edx, %ecx
100006c4b: 0f be 46 17                 	movsbl	23(%rsi), %eax
100006c4f: 0f be 57 17                 	movsbl	23(%rdi), %edx
100006c53: 0f af d0                    	imull	%eax, %edx
100006c56: 01 ca                       	addl	%ecx, %edx
100006c58: 0f be 46 18                 	movsbl	24(%rsi), %eax
100006c5c: 0f be 4f 18                 	movsbl	24(%rdi), %ecx
100006c60: 0f af c8                    	imull	%eax, %ecx
100006c63: 01 d1                       	addl	%edx, %ecx
100006c65: 0f be 46 19                 	movsbl	25(%rsi), %eax
100006c69: 0f be 57 19                 	movsbl	25(%rdi), %edx
100006c6d: 0f af d0                    	imull	%eax, %edx
100006c70: 01 ca                       	addl	%ecx, %edx
100006c72: 0f be 46 1a                 	movsbl	26(%rsi), %eax
100006c76: 0f be 4f 1a                 	movsbl	26(%rdi), %ecx
100006c7a: 0f af c8                    	imull	%eax, %ecx
100006c7d: 01 d1                       	addl	%edx, %ecx
100006c7f: 0f be 46 1b                 	movsbl	27(%rsi), %eax
100006c83: 0f be 57 1b                 	movsbl	27(%rdi), %edx
100006c87: 0f af d0                    	imull	%eax, %edx
100006c8a: 01 ca                       	addl	%ecx, %edx
100006c8c: 0f be 46 1c                 	movsbl	28(%rsi), %eax
100006c90: 0f be 4f 1c                 	movsbl	28(%rdi), %ecx
100006c94: 0f af c8                    	imull	%eax, %ecx
100006c97: 01 d1                       	addl	%edx, %ecx
100006c99: 0f be 46 1d                 	movsbl	29(%rsi), %eax
100006c9d: 0f be 57 1d                 	movsbl	29(%rdi), %edx
100006ca1: 0f af d0                    	imull	%eax, %edx
100006ca4: 01 ca                       	addl	%ecx, %edx
100006ca6: 0f be 46 1e                 	movsbl	30(%rsi), %eax
100006caa: 0f be 4f 1e                 	movsbl	30(%rdi), %ecx
100006cae: 0f af c8                    	imull	%eax, %ecx
100006cb1: 01 d1                       	addl	%edx, %ecx
100006cb3: 0f be 46 1f                 	movsbl	31(%rsi), %eax
100006cb7: 0f be 57 1f                 	movsbl	31(%rdi), %edx
100006cbb: 0f af d0                    	imull	%eax, %edx
100006cbe: 01 ca                       	addl	%ecx, %edx
100006cc0: 0f be 47 20                 	movsbl	32(%rdi), %eax
100006cc4: 0f be 4e 20                 	movsbl	32(%rsi), %ecx
100006cc8: 0f af c8                    	imull	%eax, %ecx
100006ccb: 01 d1                       	addl	%edx, %ecx
100006ccd: 0f be 47 21                 	movsbl	33(%rdi), %eax
100006cd1: 0f be 56 21                 	movsbl	33(%rsi), %edx
100006cd5: 0f af d0                    	imull	%eax, %edx
100006cd8: 01 ca                       	addl	%ecx, %edx
100006cda: 0f be 47 22                 	movsbl	34(%rdi), %eax
100006cde: 0f be 4e 22                 	movsbl	34(%rsi), %ecx
100006ce2: 0f af c8                    	imull	%eax, %ecx
100006ce5: 01 d1                       	addl	%edx, %ecx
100006ce7: 0f be 47 23                 	movsbl	35(%rdi), %eax
100006ceb: 0f be 56 23                 	movsbl	35(%rsi), %edx
100006cef: 0f af d0                    	imull	%eax, %edx
100006cf2: 01 ca                       	addl	%ecx, %edx
100006cf4: 0f be 47 24                 	movsbl	36(%rdi), %eax
100006cf8: 0f be 4e 24                 	movsbl	36(%rsi), %ecx
100006cfc: 0f af c8                    	imull	%eax, %ecx
100006cff: 01 d1                       	addl	%edx, %ecx
100006d01: 0f be 47 25                 	movsbl	37(%rdi), %eax
100006d05: 0f be 56 25                 	movsbl	37(%rsi), %edx
100006d09: 0f af d0                    	imull	%eax, %edx
100006d0c: 01 ca                       	addl	%ecx, %edx
100006d0e: 0f be 47 26                 	movsbl	38(%rdi), %eax
100006d12: 0f be 4e 26                 	movsbl	38(%rsi), %ecx
100006d16: 0f af c8                    	imull	%eax, %ecx
100006d19: 01 d1                       	addl	%edx, %ecx
100006d1b: 0f be 47 27                 	movsbl	39(%rdi), %eax
100006d1f: 0f be 56 27                 	movsbl	39(%rsi), %edx
100006d23: 0f af d0                    	imull	%eax, %edx
100006d26: 01 ca                       	addl	%ecx, %edx
100006d28: 0f be 47 28                 	movsbl	40(%rdi), %eax
100006d2c: 0f be 4e 28                 	movsbl	40(%rsi), %ecx
100006d30: 0f af c8                    	imull	%eax, %ecx
100006d33: 01 d1                       	addl	%edx, %ecx
100006d35: 0f be 47 29                 	movsbl	41(%rdi), %eax
100006d39: 0f be 56 29                 	movsbl	41(%rsi), %edx
100006d3d: 0f af d0                    	imull	%eax, %edx
100006d40: 01 ca                       	addl	%ecx, %edx
100006d42: 0f be 47 2a                 	movsbl	42(%rdi), %eax
100006d46: 0f be 4e 2a                 	movsbl	42(%rsi), %ecx
100006d4a: 0f af c8                    	imull	%eax, %ecx
100006d4d: 01 d1                       	addl	%edx, %ecx
100006d4f: 0f be 47 2b                 	movsbl	43(%rdi), %eax
100006d53: 0f be 56 2b                 	movsbl	43(%rsi), %edx
100006d57: 0f af d0                    	imull	%eax, %edx
100006d5a: 01 ca                       	addl	%ecx, %edx
100006d5c: 0f be 47 2c                 	movsbl	44(%rdi), %eax
100006d60: 0f be 4e 2c                 	movsbl	44(%rsi), %ecx
100006d64: 0f af c8                    	imull	%eax, %ecx
100006d67: 01 d1                       	addl	%edx, %ecx
100006d69: 0f be 47 2d                 	movsbl	45(%rdi), %eax
100006d6d: 0f be 56 2d                 	movsbl	45(%rsi), %edx
100006d71: 0f af d0                    	imull	%eax, %edx
100006d74: 01 ca                       	addl	%ecx, %edx
100006d76: 0f be 47 2e                 	movsbl	46(%rdi), %eax
100006d7a: 0f be 4e 2e                 	movsbl	46(%rsi), %ecx
100006d7e: 0f af c8                    	imull	%eax, %ecx
100006d81: 01 d1                       	addl	%edx, %ecx
100006d83: 0f be 57 2f                 	movsbl	47(%rdi), %edx
100006d87: 0f be 46 2f                 	movsbl	47(%rsi), %eax
100006d8b: 0f af c2                    	imull	%edx, %eax
100006d8e: 01 c8                       	addl	%ecx, %eax
100006d90: 5d                          	popq	%rbp
100006d91: c3                          	retq
100006d92: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100006d9c: 0f 1f 40 00                 	nopl	(%rax)
100006da0: 55                          	pushq	%rbp
100006da1: 48 89 e5                    	movq	%rsp, %rbp
100006da4: c4 e2 7d 21 47 08           	vpmovsxbd	8(%rdi), %ymm0
100006daa: c4 e2 7d 21 4f 18           	vpmovsxbd	24(%rdi), %ymm1
100006db0: c4 e2 7d 21 17              	vpmovsxbd	(%rdi), %ymm2
100006db5: c4 e2 7d 21 5f 10           	vpmovsxbd	16(%rdi), %ymm3
100006dbb: c4 e2 7d 21 66 08           	vpmovsxbd	8(%rsi), %ymm4
100006dc1: c4 e2 5d 40 c0              	vpmulld	%ymm0, %ymm4, %ymm0
100006dc6: c4 e2 7d 21 66 18           	vpmovsxbd	24(%rsi), %ymm4
100006dcc: c4 e2 5d 40 c9              	vpmulld	%ymm1, %ymm4, %ymm1
100006dd1: c4 e2 7d 21 26              	vpmovsxbd	(%rsi), %ymm4
100006dd6: c4 e2 5d 40 d2              	vpmulld	%ymm2, %ymm4, %ymm2
100006ddb: c4 e2 7d 21 66 10           	vpmovsxbd	16(%rsi), %ymm4
100006de1: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006de5: c4 e2 5d 40 cb              	vpmulld	%ymm3, %ymm4, %ymm1
100006dea: c5 ed fe c9                 	vpaddd	%ymm1, %ymm2, %ymm1
100006dee: c5 f5 fe c0                 	vpaddd	%ymm0, %ymm1, %ymm0
100006df2: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006df8: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006dfc: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006e01: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006e05: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006e0a: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006e0e: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006e12: 5d                          	popq	%rbp
100006e13: c5 f8 77                    	vzeroupper
100006e16: c3                          	retq

Disassembly of section __TEXT,__stubs:

0000000100006e18 __stubs:
100006e18: ff 25 e2 31 00 00           	jmpq	*12770(%rip)
100006e1e: ff 25 e4 31 00 00           	jmpq	*12772(%rip)
100006e24: ff 25 e6 31 00 00           	jmpq	*12774(%rip)
100006e2a: ff 25 e8 31 00 00           	jmpq	*12776(%rip)
100006e30: ff 25 ea 31 00 00           	jmpq	*12778(%rip)
100006e36: ff 25 ec 31 00 00           	jmpq	*12780(%rip)
100006e3c: ff 25 ee 31 00 00           	jmpq	*12782(%rip)
100006e42: ff 25 f0 31 00 00           	jmpq	*12784(%rip)
100006e48: ff 25 f2 31 00 00           	jmpq	*12786(%rip)
100006e4e: ff 25 f4 31 00 00           	jmpq	*12788(%rip)
100006e54: ff 25 f6 31 00 00           	jmpq	*12790(%rip)
100006e5a: ff 25 f8 31 00 00           	jmpq	*12792(%rip)
100006e60: ff 25 fa 31 00 00           	jmpq	*12794(%rip)
100006e66: ff 25 fc 31 00 00           	jmpq	*12796(%rip)
100006e6c: ff 25 fe 31 00 00           	jmpq	*12798(%rip)
100006e72: ff 25 00 32 00 00           	jmpq	*12800(%rip)
100006e78: ff 25 02 32 00 00           	jmpq	*12802(%rip)
100006e7e: ff 25 04 32 00 00           	jmpq	*12804(%rip)
100006e84: ff 25 06 32 00 00           	jmpq	*12806(%rip)
100006e8a: ff 25 08 32 00 00           	jmpq	*12808(%rip)
100006e90: ff 25 0a 32 00 00           	jmpq	*12810(%rip)
100006e96: ff 25 0c 32 00 00           	jmpq	*12812(%rip)
100006e9c: ff 25 0e 32 00 00           	jmpq	*12814(%rip)
100006ea2: ff 25 10 32 00 00           	jmpq	*12816(%rip)
100006ea8: ff 25 12 32 00 00           	jmpq	*12818(%rip)
100006eae: ff 25 14 32 00 00           	jmpq	*12820(%rip)
100006eb4: ff 25 16 32 00 00           	jmpq	*12822(%rip)
100006eba: ff 25 18 32 00 00           	jmpq	*12824(%rip)
100006ec0: ff 25 1a 32 00 00           	jmpq	*12826(%rip)
100006ec6: ff 25 1c 32 00 00           	jmpq	*12828(%rip)
100006ecc: ff 25 1e 32 00 00           	jmpq	*12830(%rip)
100006ed2: ff 25 20 32 00 00           	jmpq	*12832(%rip)
100006ed8: ff 25 22 32 00 00           	jmpq	*12834(%rip)
100006ede: ff 25 24 32 00 00           	jmpq	*12836(%rip)
100006ee4: ff 25 26 32 00 00           	jmpq	*12838(%rip)
100006eea: ff 25 28 32 00 00           	jmpq	*12840(%rip)
100006ef0: ff 25 2a 32 00 00           	jmpq	*12842(%rip)
100006ef6: ff 25 2c 32 00 00           	jmpq	*12844(%rip)
100006efc: ff 25 2e 32 00 00           	jmpq	*12846(%rip)

Disassembly of section __TEXT,__stub_helper:

0000000100006f04 __stub_helper:
100006f04: 4c 8d 1d 2d 32 00 00        	leaq	12845(%rip), %r11
100006f0b: 41 53                       	pushq	%r11
100006f0d: ff 25 55 21 00 00           	jmpq	*8533(%rip)
100006f13: 90                          	nop
100006f14: 68 7b 01 00 00              	pushq	$379
100006f19: e9 e6 ff ff ff              	jmp	-26 <__stub_helper>
100006f1e: 68 c9 02 00 00              	pushq	$713
100006f23: e9 dc ff ff ff              	jmp	-36 <__stub_helper>
100006f28: 68 44 00 00 00              	pushq	$68
100006f2d: e9 d2 ff ff ff              	jmp	-46 <__stub_helper>
100006f32: 68 a7 00 00 00              	pushq	$167
100006f37: e9 c8 ff ff ff              	jmp	-56 <__stub_helper>
100006f3c: 68 c8 00 00 00              	pushq	$200
100006f41: e9 be ff ff ff              	jmp	-66 <__stub_helper>
100006f46: 68 5b 03 00 00              	pushq	$859
100006f4b: e9 b4 ff ff ff              	jmp	-76 <__stub_helper>
100006f50: 68 e6 01 00 00              	pushq	$486
100006f55: e9 aa ff ff ff              	jmp	-86 <__stub_helper>
100006f5a: 68 34 02 00 00              	pushq	$564
100006f5f: e9 a0 ff ff ff              	jmp	-96 <__stub_helper>
100006f64: 68 e1 02 00 00              	pushq	$737
100006f69: e9 96 ff ff ff              	jmp	-106 <__stub_helper>
100006f6e: 68 17 00 00 00              	pushq	$23
100006f73: e9 8c ff ff ff              	jmp	-116 <__stub_helper>
100006f78: 68 f1 00 00 00              	pushq	$241
100006f7d: e9 82 ff ff ff              	jmp	-126 <__stub_helper>
100006f82: 68 12 01 00 00              	pushq	$274
100006f87: e9 78 ff ff ff              	jmp	-136 <__stub_helper>
100006f8c: 68 32 01 00 00              	pushq	$306
100006f91: e9 6e ff ff ff              	jmp	-146 <__stub_helper>
100006f96: 68 54 01 00 00              	pushq	$340
100006f9b: e9 64 ff ff ff              	jmp	-156 <__stub_helper>
100006fa0: 68 23 03 00 00              	pushq	$803
100006fa5: e9 5a ff ff ff              	jmp	-166 <__stub_helper>
100006faa: 68 3e 03 00 00              	pushq	$830
100006faf: e9 50 ff ff ff              	jmp	-176 <__stub_helper>
100006fb4: 68 85 03 00 00              	pushq	$901
100006fb9: e9 46 ff ff ff              	jmp	-186 <__stub_helper>
100006fbe: 68 b4 03 00 00              	pushq	$948
100006fc3: e9 3c ff ff ff              	jmp	-196 <__stub_helper>
100006fc8: 68 da 03 00 00              	pushq	$986
100006fcd: e9 32 ff ff ff              	jmp	-206 <__stub_helper>
100006fd2: 68 2e 04 00 00              	pushq	$1070
100006fd7: e9 28 ff ff ff              	jmp	-216 <__stub_helper>
100006fdc: 68 83 04 00 00              	pushq	$1155
100006fe1: e9 1e ff ff ff              	jmp	-226 <__stub_helper>
100006fe6: 68 d8 04 00 00              	pushq	$1240
100006feb: e9 14 ff ff ff              	jmp	-236 <__stub_helper>
100006ff0: 68 1f 05 00 00              	pushq	$1311
100006ff5: e9 0a ff ff ff              	jmp	-246 <__stub_helper>
100006ffa: 68 63 05 00 00              	pushq	$1379
100006fff: e9 00 ff ff ff              	jmp	-256 <__stub_helper>
100007004: 68 91 05 00 00              	pushq	$1425
100007009: e9 f6 fe ff ff              	jmp	-266 <__stub_helper>
10000700e: 68 af 05 00 00              	pushq	$1455
100007013: e9 ec fe ff ff              	jmp	-276 <__stub_helper>
100007018: 68 f0 05 00 00              	pushq	$1520
10000701d: e9 e2 fe ff ff              	jmp	-286 <__stub_helper>
100007022: 68 14 06 00 00              	pushq	$1556
100007027: e9 d8 fe ff ff              	jmp	-296 <__stub_helper>
10000702c: 68 33 06 00 00              	pushq	$1587
100007031: e9 ce fe ff ff              	jmp	-306 <__stub_helper>
100007036: 68 52 06 00 00              	pushq	$1618
10000703b: e9 c4 fe ff ff              	jmp	-316 <__stub_helper>
100007040: 68 6b 06 00 00              	pushq	$1643
100007045: e9 ba fe ff ff              	jmp	-326 <__stub_helper>
10000704a: 68 86 06 00 00              	pushq	$1670
10000704f: e9 b0 fe ff ff              	jmp	-336 <__stub_helper>
100007054: 68 00 00 00 00              	pushq	$0
100007059: e9 a6 fe ff ff              	jmp	-346 <__stub_helper>
10000705e: 68 9f 06 00 00              	pushq	$1695
100007063: e9 9c fe ff ff              	jmp	-356 <__stub_helper>
100007068: 68 b9 06 00 00              	pushq	$1721
10000706d: e9 92 fe ff ff              	jmp	-366 <__stub_helper>
100007072: 68 c9 06 00 00              	pushq	$1737
100007077: e9 88 fe ff ff              	jmp	-376 <__stub_helper>
