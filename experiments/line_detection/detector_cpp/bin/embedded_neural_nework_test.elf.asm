
bin/embedded_neural_nework_test.elf:	file format Mach-O 64-bit x86-64


Disassembly of section __TEXT,__text:

0000000100002680 __Z8get_timev:
100002680: 55                          	pushq	%rbp
100002681: 48 89 e5                    	movq	%rsp, %rbp
100002684: e8 e9 47 00 00              	callq	18409 <dyld_stub_binder+0x100006e72>
100002689: c4 e1 fb 2a c0              	vcvtsi2sd	%rax, %xmm0, %xmm0
10000268e: c5 fb 5e 05 aa 49 00 00     	vdivsd	18858(%rip), %xmm0, %xmm0
100002696: 5d                          	popq	%rbp
100002697: c3                          	retq
100002698: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

00000001000026a0 __Z14get_predictionRN2cv3MatER14ModelInterfacef:
1000026a0: 55                          	pushq	%rbp
1000026a1: 48 89 e5                    	movq	%rsp, %rbp
1000026a4: 41 57                       	pushq	%r15
1000026a6: 41 56                       	pushq	%r14
1000026a8: 41 55                       	pushq	%r13
1000026aa: 41 54                       	pushq	%r12
1000026ac: 53                          	pushq	%rbx
1000026ad: 48 81 ec 28 01 00 00        	subq	$296, %rsp
1000026b4: c5 fa 11 45 a8              	vmovss	%xmm0, -88(%rbp)
1000026b9: 49 89 d6                    	movq	%rdx, %r14
1000026bc: 49 89 f4                    	movq	%rsi, %r12
1000026bf: 48 89 fb                    	movq	%rdi, %rbx
1000026c2: 48 8b 05 97 69 00 00        	movq	27031(%rip), %rax
1000026c9: 48 8b 00                    	movq	(%rax), %rax
1000026cc: 48 89 45 d0                 	movq	%rax, -48(%rbp)
1000026d0: 8b 46 08                    	movl	8(%rsi), %eax
1000026d3: 8b 4e 0c                    	movl	12(%rsi), %ecx
1000026d6: c7 85 d0 fe ff ff 00 00 ff 42       	movl	$1124007936, -304(%rbp)
1000026e0: 48 8d 95 d8 fe ff ff        	leaq	-296(%rbp), %rdx
1000026e7: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000026eb: c5 fc 11 85 d4 fe ff ff     	vmovups	%ymm0, -300(%rbp)
1000026f3: c5 fc 11 85 f0 fe ff ff     	vmovups	%ymm0, -272(%rbp)
1000026fb: 48 89 95 10 ff ff ff        	movq	%rdx, -240(%rbp)
100002702: 48 8d 95 20 ff ff ff        	leaq	-224(%rbp), %rdx
100002709: 48 89 95 18 ff ff ff        	movq	%rdx, -232(%rbp)
100002710: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002714: c5 f8 11 85 20 ff ff ff     	vmovups	%xmm0, -224(%rbp)
10000271c: 89 4d b8                    	movl	%ecx, -72(%rbp)
10000271f: 89 45 bc                    	movl	%eax, -68(%rbp)
100002722: 4c 8d bd d0 fe ff ff        	leaq	-304(%rbp), %r15
100002729: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
10000272d: 4c 89 ff                    	movq	%r15, %rdi
100002730: be 02 00 00 00              	movl	$2, %esi
100002735: 31 c9                       	xorl	%ecx, %ecx
100002737: c5 f8 77                    	vzeroupper
10000273a: e8 c7 46 00 00              	callq	18119 <dyld_stub_binder+0x100006e06>
10000273f: 48 c7 85 40 ff ff ff 00 00 00 00    	movq	$0, -192(%rbp)
10000274a: c7 85 30 ff ff ff 00 00 01 01       	movl	$16842752, -208(%rbp)
100002754: 4c 89 a5 38 ff ff ff        	movq	%r12, -200(%rbp)
10000275b: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
100002763: c7 45 b8 00 00 01 02        	movl	$33619968, -72(%rbp)
10000276a: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
10000276e: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002775: 48 8d 75 b8                 	leaq	-72(%rbp), %rsi
100002779: ba 06 00 00 00              	movl	$6, %edx
10000277e: 31 c9                       	xorl	%ecx, %ecx
100002780: e8 ab 46 00 00              	callq	18091 <dyld_stub_binder+0x100006e30>
100002785: 41 8b 44 24 08              	movl	8(%r12), %eax
10000278a: 41 8b 4c 24 0c              	movl	12(%r12), %ecx
10000278f: c7 85 30 ff ff ff 00 00 ff 42       	movl	$1124007936, -208(%rbp)
100002799: 48 8d 95 38 ff ff ff        	leaq	-200(%rbp), %rdx
1000027a0: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000027a4: c5 fc 11 85 34 ff ff ff     	vmovups	%ymm0, -204(%rbp)
1000027ac: c5 fc 11 85 50 ff ff ff     	vmovups	%ymm0, -176(%rbp)
1000027b4: 48 89 95 70 ff ff ff        	movq	%rdx, -144(%rbp)
1000027bb: 48 8d 55 80                 	leaq	-128(%rbp), %rdx
1000027bf: 48 89 95 78 ff ff ff        	movq	%rdx, -136(%rbp)
1000027c6: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000027ca: c5 f8 11 45 80              	vmovups	%xmm0, -128(%rbp)
1000027cf: 89 4d b8                    	movl	%ecx, -72(%rbp)
1000027d2: 89 45 bc                    	movl	%eax, -68(%rbp)
1000027d5: 4c 8d a5 30 ff ff ff        	leaq	-208(%rbp), %r12
1000027dc: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
1000027e0: 4c 89 e7                    	movq	%r12, %rdi
1000027e3: be 02 00 00 00              	movl	$2, %esi
1000027e8: 31 c9                       	xorl	%ecx, %ecx
1000027ea: c5 f8 77                    	vzeroupper
1000027ed: e8 14 46 00 00              	callq	17940 <dyld_stub_binder+0x100006e06>
1000027f2: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
1000027fa: c7 45 b8 00 00 01 01        	movl	$16842752, -72(%rbp)
100002801: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
100002805: 48 c7 85 c0 fe ff ff 00 00 00 00    	movq	$0, -320(%rbp)
100002810: c7 85 b0 fe ff ff 00 00 01 02       	movl	$33619968, -336(%rbp)
10000281a: 4c 89 a5 b8 fe ff ff        	movq	%r12, -328(%rbp)
100002821: 41 8b 46 0c                 	movl	12(%r14), %eax
100002825: 41 8b 4e 10                 	movl	16(%r14), %ecx
100002829: 89 4d 90                    	movl	%ecx, -112(%rbp)
10000282c: 89 45 94                    	movl	%eax, -108(%rbp)
10000282f: 48 8d 7d b8                 	leaq	-72(%rbp), %rdi
100002833: 48 8d b5 b0 fe ff ff        	leaq	-336(%rbp), %rsi
10000283a: 48 8d 55 90                 	leaq	-112(%rbp), %rdx
10000283e: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002842: c5 f0 57 c9                 	vxorps	%xmm1, %xmm1, %xmm1
100002846: b9 03 00 00 00              	movl	$3, %ecx
10000284b: e8 ce 45 00 00              	callq	17870 <dyld_stub_binder+0x100006e1e>
100002850: 41 8b 46 0c                 	movl	12(%r14), %eax
100002854: 85 c0                       	testl	%eax, %eax
100002856: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
10000285a: 4d 89 f7                    	movq	%r14, %r15
10000285d: 0f 84 7c 00 00 00           	je	124 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x23f>
100002863: 41 8b 4f 10                 	movl	16(%r15), %ecx
100002867: 31 d2                       	xorl	%edx, %edx
100002869: 45 31 e4                    	xorl	%r12d, %r12d
10000286c: 85 c9                       	testl	%ecx, %ecx
10000286e: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1dc>
100002870: 31 c9                       	xorl	%ecx, %ecx
100002872: ff c2                       	incl	%edx
100002874: 39 c2                       	cmpl	%eax, %edx
100002876: 73 67                       	jae	103 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x23f>
100002878: 85 c9                       	testl	%ecx, %ecx
10000287a: 74 f4                       	je	-12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1d0>
10000287c: 89 55 a0                    	movl	%edx, -96(%rbp)
10000287f: 4c 63 f2                    	movslq	%edx, %r14
100002882: 45 31 ed                    	xorl	%r13d, %r13d
100002885: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000288f: 90                          	nop
100002890: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
100002897: 48 8b 00                    	movq	(%rax), %rax
10000289a: 49 0f af c6                 	imulq	%r14, %rax
10000289e: 48 03 85 40 ff ff ff        	addq	-192(%rbp), %rax
1000028a5: 49 63 cd                    	movslq	%r13d, %rcx
1000028a8: 0f b6 1c 01                 	movzbl	(%rcx,%rax), %ebx
1000028ac: 4c 89 ff                    	movq	%r15, %rdi
1000028af: e8 4c 21 00 00              	callq	8524 <__ZN14ModelInterface12input_bufferEv>
1000028b4: 43 8d 0c 2c                 	leal	(%r12,%r13), %ecx
1000028b8: d0 eb                       	shrb	%bl
1000028ba: 89 c9                       	movl	%ecx, %ecx
1000028bc: 88 1c 08                    	movb	%bl, (%rax,%rcx)
1000028bf: 41 ff c5                    	incl	%r13d
1000028c2: 41 8b 4f 10                 	movl	16(%r15), %ecx
1000028c6: 41 39 cd                    	cmpl	%ecx, %r13d
1000028c9: 72 c5                       	jb	-59 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1f0>
1000028cb: 41 8b 47 0c                 	movl	12(%r15), %eax
1000028cf: 45 01 ec                    	addl	%r13d, %r12d
1000028d2: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
1000028d6: 8b 55 a0                    	movl	-96(%rbp), %edx
1000028d9: ff c2                       	incl	%edx
1000028db: 39 c2                       	cmpl	%eax, %edx
1000028dd: 72 99                       	jb	-103 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1d8>
1000028df: 49 8b 07                    	movq	(%r15), %rax
1000028e2: 4c 89 ff                    	movq	%r15, %rdi
1000028e5: ff 50 10                    	callq	*16(%rax)
1000028e8: 41 8b 47 18                 	movl	24(%r15), %eax
1000028ec: 41 8b 4f 1c                 	movl	28(%r15), %ecx
1000028f0: c7 03 00 00 ff 42           	movl	$1124007936, (%rbx)
1000028f6: 48 8d 53 08                 	leaq	8(%rbx), %rdx
1000028fa: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000028fe: c5 fc 11 43 04              	vmovups	%ymm0, 4(%rbx)
100002903: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
100002908: 48 89 53 40                 	movq	%rdx, 64(%rbx)
10000290c: 48 8d 53 50                 	leaq	80(%rbx), %rdx
100002910: 48 89 95 c8 fe ff ff        	movq	%rdx, -312(%rbp)
100002917: 48 89 53 48                 	movq	%rdx, 72(%rbx)
10000291b: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000291f: c5 f8 11 43 50              	vmovups	%xmm0, 80(%rbx)
100002924: 89 4d b8                    	movl	%ecx, -72(%rbp)
100002927: 89 45 bc                    	movl	%eax, -68(%rbp)
10000292a: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
10000292e: 48 89 df                    	movq	%rbx, %rdi
100002931: be 02 00 00 00              	movl	$2, %esi
100002936: 31 c9                       	xorl	%ecx, %ecx
100002938: c5 f8 77                    	vzeroupper
10000293b: e8 c6 44 00 00              	callq	17606 <dyld_stub_binder+0x100006e06>
100002940: 41 8b 47 18                 	movl	24(%r15), %eax
100002944: 41 83 7f 14 01              	cmpl	$1, 20(%r15)
100002949: 4d 89 fc                    	movq	%r15, %r12
10000294c: 0f 85 c7 00 00 00           	jne	199 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x379>
100002952: 85 c0                       	testl	%eax, %eax
100002954: 0f 84 e2 01 00 00           	je	482 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
10000295a: c5 fa 10 45 a8              	vmovss	-88(%rbp), %xmm0
10000295f: c5 fa 59 05 21 47 00 00     	vmulss	18209(%rip), %xmm0, %xmm0
100002967: c5 fa 11 45 a0              	vmovss	%xmm0, -96(%rbp)
10000296c: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002971: 45 31 ff                    	xorl	%r15d, %r15d
100002974: 31 d2                       	xorl	%edx, %edx
100002976: 45 31 ed                    	xorl	%r13d, %r13d
100002979: 85 c9                       	testl	%ecx, %ecx
10000297b: 75 13                       	jne	19 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2f0>
10000297d: 0f 1f 00                    	nopl	(%rax)
100002980: 31 c9                       	xorl	%ecx, %ecx
100002982: ff c2                       	incl	%edx
100002984: 39 c2                       	cmpl	%eax, %edx
100002986: 0f 83 b0 01 00 00           	jae	432 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
10000298c: 85 c9                       	testl	%ecx, %ecx
10000298e: 74 f0                       	je	-16 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2e0>
100002990: 89 55 a8                    	movl	%edx, -88(%rbp)
100002993: 4c 63 f2                    	movslq	%edx, %r14
100002996: 31 db                       	xorl	%ebx, %ebx
100002998: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
1000029a0: 4c 89 e7                    	movq	%r12, %rdi
1000029a3: e8 68 20 00 00              	callq	8296 <__ZN14ModelInterface13output_bufferEv>
1000029a8: 42 8d 0c 2b                 	leal	(%rbx,%r13), %ecx
1000029ac: 89 c9                       	movl	%ecx, %ecx
1000029ae: 0f b6 04 08                 	movzbl	(%rax,%rcx), %eax
1000029b2: 84 c0                       	testb	%al, %al
1000029b4: 41 0f 48 c7                 	cmovsl	%r15d, %eax
1000029b8: 0f be c8                    	movsbl	%al, %ecx
1000029bb: c5 ea 2a c1                 	vcvtsi2ss	%ecx, %xmm2, %xmm0
1000029bf: 48 8b 55 b0                 	movq	-80(%rbp), %rdx
1000029c3: 48 8b 4a 48                 	movq	72(%rdx), %rcx
1000029c7: 48 8b 09                    	movq	(%rcx), %rcx
1000029ca: 49 0f af ce                 	imulq	%r14, %rcx
1000029ce: 48 03 4a 10                 	addq	16(%rdx), %rcx
1000029d2: 48 63 db                    	movslq	%ebx, %rbx
1000029d5: 88 04 0b                    	movb	%al, (%rbx,%rcx)
1000029d8: 48 8b 42 48                 	movq	72(%rdx), %rax
1000029dc: 48 8b 00                    	movq	(%rax), %rax
1000029df: 49 0f af c6                 	imulq	%r14, %rax
1000029e3: 48 03 42 10                 	addq	16(%rdx), %rax
1000029e7: c5 f8 2e 45 a0              	vucomiss	-96(%rbp), %xmm0
1000029ec: 0f 97 04 03                 	seta	(%rbx,%rax)
1000029f0: ff c3                       	incl	%ebx
1000029f2: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
1000029f7: 39 cb                       	cmpl	%ecx, %ebx
1000029f9: 72 a5                       	jb	-91 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x300>
1000029fb: 41 8b 44 24 18              	movl	24(%r12), %eax
100002a00: 41 01 dd                    	addl	%ebx, %r13d
100002a03: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002a07: 8b 55 a8                    	movl	-88(%rbp), %edx
100002a0a: ff c2                       	incl	%edx
100002a0c: 39 c2                       	cmpl	%eax, %edx
100002a0e: 0f 82 78 ff ff ff           	jb	-136 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2ec>
100002a14: e9 23 01 00 00              	jmp	291 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002a19: 85 c0                       	testl	%eax, %eax
100002a1b: 0f 84 1b 01 00 00           	je	283 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002a21: c5 fa 10 45 a8              	vmovss	-88(%rbp), %xmm0
100002a26: c5 fa 59 05 5a 46 00 00     	vmulss	18010(%rip), %xmm0, %xmm0
100002a2e: c5 fa 11 45 98              	vmovss	%xmm0, -104(%rbp)
100002a33: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002a38: 31 d2                       	xorl	%edx, %edx
100002a3a: 45 31 ff                    	xorl	%r15d, %r15d
100002a3d: 85 c9                       	testl	%ecx, %ecx
100002a3f: 75 29                       	jne	41 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3ca>
100002a41: e9 ea 00 00 00              	jmp	234 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x490>
100002a46: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002a50: 41 8b 44 24 18              	movl	24(%r12), %eax
100002a55: 8b 55 9c                    	movl	-100(%rbp), %edx
100002a58: ff c2                       	incl	%edx
100002a5a: 39 c2                       	cmpl	%eax, %edx
100002a5c: 0f 83 da 00 00 00           	jae	218 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002a62: 85 c9                       	testl	%ecx, %ecx
100002a64: 0f 84 c6 00 00 00           	je	198 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x490>
100002a6a: 89 55 9c                    	movl	%edx, -100(%rbp)
100002a6d: 48 63 c2                    	movslq	%edx, %rax
100002a70: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100002a74: 31 d2                       	xorl	%edx, %edx
100002a76: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100002a7a: 41 83 7c 24 14 00           	cmpl	$0, 20(%r12)
100002a80: 75 60                       	jne	96 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x442>
100002a82: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002a8c: 0f 1f 40 00                 	nopl	(%rax)
100002a90: 41 b6 81                    	movb	$-127, %r14b
100002a93: 45 31 ed                    	xorl	%r13d, %r13d
100002a96: 41 0f be c6                 	movsbl	%r14b, %eax
100002a9a: c5 ea 2a c0                 	vcvtsi2ss	%eax, %xmm2, %xmm0
100002a9e: c5 f8 2e 45 98              	vucomiss	-104(%rbp), %xmm0
100002aa3: b8 00 00 00 00              	movl	$0, %eax
100002aa8: 44 0f 46 e8                 	cmovbel	%eax, %r13d
100002aac: 48 8b 43 48                 	movq	72(%rbx), %rax
100002ab0: 48 8b 00                    	movq	(%rax), %rax
100002ab3: 48 0f af 45 a8              	imulq	-88(%rbp), %rax
100002ab8: 48 03 43 10                 	addq	16(%rbx), %rax
100002abc: 48 8b 55 a0                 	movq	-96(%rbp), %rdx
100002ac0: 48 63 d2                    	movslq	%edx, %rdx
100002ac3: 44 88 2c 02                 	movb	%r13b, (%rdx,%rax)
100002ac7: ff c2                       	incl	%edx
100002ac9: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002ace: 39 ca                       	cmpl	%ecx, %edx
100002ad0: 0f 83 7a ff ff ff           	jae	-134 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3b0>
100002ad6: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100002ada: 41 83 7c 24 14 00           	cmpl	$0, 20(%r12)
100002ae0: 74 ae                       	je	-82 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3f0>
100002ae2: 41 b6 81                    	movb	$-127, %r14b
100002ae5: 31 db                       	xorl	%ebx, %ebx
100002ae7: 45 31 ed                    	xorl	%r13d, %r13d
100002aea: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100002af0: 4c 89 e7                    	movq	%r12, %rdi
100002af3: e8 18 1f 00 00              	callq	7960 <__ZN14ModelInterface13output_bufferEv>
100002af8: 41 8d 0c 1f                 	leal	(%r15,%rbx), %ecx
100002afc: 89 c9                       	movl	%ecx, %ecx
100002afe: 0f b6 04 08                 	movzbl	(%rax,%rcx), %eax
100002b02: 44 38 f0                    	cmpb	%r14b, %al
100002b05: 44 0f 4f eb                 	cmovgl	%ebx, %r13d
100002b09: 45 0f b6 f6                 	movzbl	%r14b, %r14d
100002b0d: 44 0f 4d f0                 	cmovgel	%eax, %r14d
100002b11: ff c3                       	incl	%ebx
100002b13: 41 3b 5c 24 14              	cmpl	20(%r12), %ebx
100002b18: 72 d6                       	jb	-42 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x450>
100002b1a: 41 01 df                    	addl	%ebx, %r15d
100002b1d: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002b21: e9 70 ff ff ff              	jmp	-144 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3f6>
100002b26: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002b30: 31 c9                       	xorl	%ecx, %ecx
100002b32: ff c2                       	incl	%edx
100002b34: 39 c2                       	cmpl	%eax, %edx
100002b36: 0f 82 26 ff ff ff           	jb	-218 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3c2>
100002b3c: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002b43: 48 85 c0                    	testq	%rax, %rax
100002b46: 74 12                       	je	18 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4ba>
100002b48: f0                          	lock
100002b49: ff 48 14                    	decl	20(%rax)
100002b4c: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4ba>
100002b4e: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002b55: e8 a6 42 00 00              	callq	17062 <dyld_stub_binder+0x100006e00>
100002b5a: 48 c7 85 68 ff ff ff 00 00 00 00    	movq	$0, -152(%rbp)
100002b65: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002b69: c5 fc 11 85 40 ff ff ff     	vmovups	%ymm0, -192(%rbp)
100002b71: 83 bd 34 ff ff ff 00        	cmpl	$0, -204(%rbp)
100002b78: 7e 2c                       	jle	44 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x506>
100002b7a: 48 8b 85 70 ff ff ff        	movq	-144(%rbp), %rax
100002b81: 31 c9                       	xorl	%ecx, %ecx
100002b83: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002b8d: 0f 1f 00                    	nopl	(%rax)
100002b90: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002b97: 48 ff c1                    	incq	%rcx
100002b9a: 48 63 95 34 ff ff ff        	movslq	-204(%rbp), %rdx
100002ba1: 48 39 d1                    	cmpq	%rdx, %rcx
100002ba4: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4f0>
100002ba6: 48 8b bd 78 ff ff ff        	movq	-136(%rbp), %rdi
100002bad: 48 8d 45 80                 	leaq	-128(%rbp), %rax
100002bb1: 48 39 c7                    	cmpq	%rax, %rdi
100002bb4: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x51e>
100002bb6: c5 f8 77                    	vzeroupper
100002bb9: e8 78 42 00 00              	callq	17016 <dyld_stub_binder+0x100006e36>
100002bbe: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002bc5: 48 85 c0                    	testq	%rax, %rax
100002bc8: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x53f>
100002bca: f0                          	lock
100002bcb: ff 48 14                    	decl	20(%rax)
100002bce: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x53f>
100002bd0: 48 8d bd d0 fe ff ff        	leaq	-304(%rbp), %rdi
100002bd7: c5 f8 77                    	vzeroupper
100002bda: e8 21 42 00 00              	callq	16929 <dyld_stub_binder+0x100006e00>
100002bdf: 48 c7 85 08 ff ff ff 00 00 00 00    	movq	$0, -248(%rbp)
100002bea: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002bee: c5 fc 11 85 e0 fe ff ff     	vmovups	%ymm0, -288(%rbp)
100002bf6: 83 bd d4 fe ff ff 00        	cmpl	$0, -300(%rbp)
100002bfd: 7e 27                       	jle	39 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x586>
100002bff: 48 8b 85 10 ff ff ff        	movq	-240(%rbp), %rax
100002c06: 31 c9                       	xorl	%ecx, %ecx
100002c08: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002c10: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002c17: 48 ff c1                    	incq	%rcx
100002c1a: 48 63 95 d4 fe ff ff        	movslq	-300(%rbp), %rdx
100002c21: 48 39 d1                    	cmpq	%rdx, %rcx
100002c24: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x570>
100002c26: 48 8b bd 18 ff ff ff        	movq	-232(%rbp), %rdi
100002c2d: 48 8d 85 20 ff ff ff        	leaq	-224(%rbp), %rax
100002c34: 48 39 c7                    	cmpq	%rax, %rdi
100002c37: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5a1>
100002c39: c5 f8 77                    	vzeroupper
100002c3c: e8 f5 41 00 00              	callq	16885 <dyld_stub_binder+0x100006e36>
100002c41: 48 8b 05 18 64 00 00        	movq	25624(%rip), %rax
100002c48: 48 8b 00                    	movq	(%rax), %rax
100002c4b: 48 3b 45 d0                 	cmpq	-48(%rbp), %rax
100002c4f: 75 18                       	jne	24 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5c9>
100002c51: 48 89 d8                    	movq	%rbx, %rax
100002c54: 48 81 c4 28 01 00 00        	addq	$296, %rsp
100002c5b: 5b                          	popq	%rbx
100002c5c: 41 5c                       	popq	%r12
100002c5e: 41 5d                       	popq	%r13
100002c60: 41 5e                       	popq	%r14
100002c62: 41 5f                       	popq	%r15
100002c64: 5d                          	popq	%rbp
100002c65: c5 f8 77                    	vzeroupper
100002c68: c3                          	retq
100002c69: c5 f8 77                    	vzeroupper
100002c6c: e8 49 42 00 00              	callq	16969 <dyld_stub_binder+0x100006eba>
100002c71: 48 89 c7                    	movq	%rax, %rdi
100002c74: e8 e7 16 00 00              	callq	5863 <_main+0x1510>
100002c79: 48 89 c7                    	movq	%rax, %rdi
100002c7c: e8 df 16 00 00              	callq	5855 <_main+0x1510>
100002c81: eb 1e                       	jmp	30 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x601>
100002c83: eb 00                       	jmp	0 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5e5>
100002c85: 49 89 c6                    	movq	%rax, %r14
100002c88: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002c8f: 48 85 c0                    	testq	%rax, %rax
100002c92: 0f 85 0f 01 00 00           	jne	271 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x707>
100002c98: e9 1f 01 00 00              	jmp	287 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002c9d: eb 02                       	jmp	2 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x601>
100002c9f: eb 14                       	jmp	20 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x615>
100002ca1: 49 89 c6                    	movq	%rax, %r14
100002ca4: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002cab: 48 85 c0                    	testq	%rax, %rax
100002cae: 75 7f                       	jne	127 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x68f>
100002cb0: e9 8f 00 00 00              	jmp	143 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002cb5: 49 89 c6                    	movq	%rax, %r14
100002cb8: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002cbc: 48 8b 43 38                 	movq	56(%rbx), %rax
100002cc0: 48 85 c0                    	testq	%rax, %rax
100002cc3: 74 0e                       	je	14 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x633>
100002cc5: f0                          	lock
100002cc6: ff 48 14                    	decl	20(%rax)
100002cc9: 75 08                       	jne	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x633>
100002ccb: 48 89 df                    	movq	%rbx, %rdi
100002cce: e8 2d 41 00 00              	callq	16685 <dyld_stub_binder+0x100006e00>
100002cd3: 48 c7 43 38 00 00 00 00     	movq	$0, 56(%rbx)
100002cdb: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002cdf: c5 fc 11 43 10              	vmovups	%ymm0, 16(%rbx)
100002ce4: 83 7b 04 00                 	cmpl	$0, 4(%rbx)
100002ce8: 7e 20                       	jle	32 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x66a>
100002cea: 48 8b 4d b0                 	movq	-80(%rbp), %rcx
100002cee: 48 8d 41 04                 	leaq	4(%rcx), %rax
100002cf2: 48 8b 49 40                 	movq	64(%rcx), %rcx
100002cf6: 31 d2                       	xorl	%edx, %edx
100002cf8: c7 04 91 00 00 00 00        	movl	$0, (%rcx,%rdx,4)
100002cff: 48 ff c2                    	incq	%rdx
100002d02: 48 63 30                    	movslq	(%rax), %rsi
100002d05: 48 39 f2                    	cmpq	%rsi, %rdx
100002d08: 7c ee                       	jl	-18 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x658>
100002d0a: 48 8b 45 b0                 	movq	-80(%rbp), %rax
100002d0e: 48 8b 78 48                 	movq	72(%rax), %rdi
100002d12: 48 3b bd c8 fe ff ff        	cmpq	-312(%rbp), %rdi
100002d19: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x683>
100002d1b: c5 f8 77                    	vzeroupper
100002d1e: e8 13 41 00 00              	callq	16659 <dyld_stub_binder+0x100006e36>
100002d23: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002d2a: 48 85 c0                    	testq	%rax, %rax
100002d2d: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002d2f: f0                          	lock
100002d30: ff 48 14                    	decl	20(%rax)
100002d33: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002d35: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002d3c: c5 f8 77                    	vzeroupper
100002d3f: e8 bc 40 00 00              	callq	16572 <dyld_stub_binder+0x100006e00>
100002d44: 48 c7 85 68 ff ff ff 00 00 00 00    	movq	$0, -152(%rbp)
100002d4f: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002d53: c5 fc 11 85 40 ff ff ff     	vmovups	%ymm0, -192(%rbp)
100002d5b: 83 bd 34 ff ff ff 00        	cmpl	$0, -204(%rbp)
100002d62: 7e 1f                       	jle	31 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6e3>
100002d64: 48 8b 85 70 ff ff ff        	movq	-144(%rbp), %rax
100002d6b: 31 c9                       	xorl	%ecx, %ecx
100002d6d: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002d74: 48 ff c1                    	incq	%rcx
100002d77: 48 63 95 34 ff ff ff        	movslq	-204(%rbp), %rdx
100002d7e: 48 39 d1                    	cmpq	%rdx, %rcx
100002d81: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6cd>
100002d83: 48 8b bd 78 ff ff ff        	movq	-136(%rbp), %rdi
100002d8a: 48 8d 45 80                 	leaq	-128(%rbp), %rax
100002d8e: 48 39 c7                    	cmpq	%rax, %rdi
100002d91: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6fb>
100002d93: c5 f8 77                    	vzeroupper
100002d96: e8 9b 40 00 00              	callq	16539 <dyld_stub_binder+0x100006e36>
100002d9b: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002da2: 48 85 c0                    	testq	%rax, %rax
100002da5: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002da7: f0                          	lock
100002da8: ff 48 14                    	decl	20(%rax)
100002dab: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002dad: 48 8d bd d0 fe ff ff        	leaq	-304(%rbp), %rdi
100002db4: c5 f8 77                    	vzeroupper
100002db7: e8 44 40 00 00              	callq	16452 <dyld_stub_binder+0x100006e00>
100002dbc: 48 c7 85 08 ff ff ff 00 00 00 00    	movq	$0, -248(%rbp)
100002dc7: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002dcb: c5 fc 11 85 e0 fe ff ff     	vmovups	%ymm0, -288(%rbp)
100002dd3: 83 bd d4 fe ff ff 00        	cmpl	$0, -300(%rbp)
100002dda: 7e 2a                       	jle	42 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x766>
100002ddc: 48 8b 85 10 ff ff ff        	movq	-240(%rbp), %rax
100002de3: 31 c9                       	xorl	%ecx, %ecx
100002de5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002def: 90                          	nop
100002df0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002df7: 48 ff c1                    	incq	%rcx
100002dfa: 48 63 95 d4 fe ff ff        	movslq	-300(%rbp), %rdx
100002e01: 48 39 d1                    	cmpq	%rdx, %rcx
100002e04: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x750>
100002e06: 48 8b bd 18 ff ff ff        	movq	-232(%rbp), %rdi
100002e0d: 48 8d 85 20 ff ff ff        	leaq	-224(%rbp), %rax
100002e14: 48 39 c7                    	cmpq	%rax, %rdi
100002e17: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x781>
100002e19: c5 f8 77                    	vzeroupper
100002e1c: e8 15 40 00 00              	callq	16405 <dyld_stub_binder+0x100006e36>
100002e21: 4c 89 f7                    	movq	%r14, %rdi
100002e24: c5 f8 77                    	vzeroupper
100002e27: e8 bc 3f 00 00              	callq	16316 <dyld_stub_binder+0x100006de8>
100002e2c: 0f 0b                       	ud2
100002e2e: 48 89 c7                    	movq	%rax, %rdi
100002e31: e8 2a 15 00 00              	callq	5418 <_main+0x1510>
100002e36: 48 89 c7                    	movq	%rax, %rdi
100002e39: e8 22 15 00 00              	callq	5410 <_main+0x1510>
100002e3e: 48 89 c7                    	movq	%rax, %rdi
100002e41: e8 1a 15 00 00              	callq	5402 <_main+0x1510>
100002e46: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

0000000100002e50 _main:
100002e50: 55                          	pushq	%rbp
100002e51: 48 89 e5                    	movq	%rsp, %rbp
100002e54: 41 57                       	pushq	%r15
100002e56: 41 56                       	pushq	%r14
100002e58: 41 55                       	pushq	%r13
100002e5a: 41 54                       	pushq	%r12
100002e5c: 53                          	pushq	%rbx
100002e5d: 48 83 e4 e0                 	andq	$-32, %rsp
100002e61: 48 81 ec 00 04 00 00        	subq	$1024, %rsp
100002e68: 48 8b 05 f1 61 00 00        	movq	25073(%rip), %rax
100002e6f: 48 8b 00                    	movq	(%rax), %rax
100002e72: 48 89 84 24 e0 03 00 00     	movq	%rax, 992(%rsp)
100002e7a: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100002e82: e8 d9 27 00 00              	callq	10201 <__ZN11LineNetworkC1Ev>
100002e87: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002e8b: c5 f9 7f 84 24 60 02 00 00  	vmovdqa	%xmm0, 608(%rsp)
100002e94: 48 c7 84 24 70 02 00 00 00 00 00 00 	movq	$0, 624(%rsp)
100002ea0: bf 30 00 00 00              	movl	$48, %edi
100002ea5: e8 fe 3f 00 00              	callq	16382 <dyld_stub_binder+0x100006ea8>
100002eaa: 48 89 84 24 70 02 00 00     	movq	%rax, 624(%rsp)
100002eb2: c5 f8 28 05 f6 41 00 00     	vmovaps	16886(%rip), %xmm0
100002eba: c5 f8 29 84 24 60 02 00 00  	vmovaps	%xmm0, 608(%rsp)
100002ec3: c5 fe 6f 05 e9 5f 00 00     	vmovdqu	24553(%rip), %ymm0
100002ecb: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002ecf: 48 b9 69 64 65 6f 2e 6d 70 34       	movabsq	$3778640133568685161, %rcx
100002ed9: 48 89 48 20                 	movq	%rcx, 32(%rax)
100002edd: c6 40 28 00                 	movb	$0, 40(%rax)
100002ee1: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100002ee9: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
100002ef1: 31 d2                       	xorl	%edx, %edx
100002ef3: c5 f8 77                    	vzeroupper
100002ef6: e8 f3 3e 00 00              	callq	16115 <dyld_stub_binder+0x100006dee>
100002efb: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100002f03: 74 0d                       	je	13 <_main+0xc2>
100002f05: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
100002f0d: e8 8a 3f 00 00              	callq	16266 <dyld_stub_binder+0x100006e9c>
100002f12: 4c 8d 74 24 68              	leaq	104(%rsp), %r14
100002f17: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002f1b: c5 f9 d6 84 24 d8 00 00 00  	vmovq	%xmm0, 216(%rsp)
100002f24: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100002f2c: 4c 8d bc 24 c0 03 00 00     	leaq	960(%rsp), %r15
100002f34: 4c 8d ac 24 c0 01 00 00     	leaq	448(%rsp), %r13
100002f3c: eb 0b                       	jmp	11 <_main+0xf9>
100002f3e: 66 90                       	nop
100002f40: 45 85 e4                    	testl	%r12d, %r12d
100002f43: 0f 85 a1 0f 00 00           	jne	4001 <_main+0x109a>
100002f49: 48 89 df                    	movq	%rbx, %rdi
100002f4c: c5 f8 77                    	vzeroupper
100002f4f: e8 ee 3e 00 00              	callq	16110 <dyld_stub_binder+0x100006e42>
100002f54: 84 c0                       	testb	%al, %al
100002f56: 0f 84 8e 0f 00 00           	je	3982 <_main+0x109a>
100002f5c: c7 44 24 18 00 00 ff 42     	movl	$1124007936, 24(%rsp)
100002f64: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002f68: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
100002f6d: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
100002f72: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002f76: 48 8d 44 24 20              	leaq	32(%rsp), %rax
100002f7b: 48 89 44 24 58              	movq	%rax, 88(%rsp)
100002f80: 4c 89 74 24 60              	movq	%r14, 96(%rsp)
100002f85: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002f89: c4 c1 7a 7f 06              	vmovdqu	%xmm0, (%r14)
100002f8e: 48 89 df                    	movq	%rbx, %rdi
100002f91: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100002f96: c5 f8 77                    	vzeroupper
100002f99: e8 5c 3e 00 00              	callq	15964 <dyld_stub_binder+0x100006dfa>
100002f9e: 41 bc 03 00 00 00           	movl	$3, %r12d
100002fa4: 48 83 7c 24 28 00           	cmpq	$0, 40(%rsp)
100002faa: 0f 84 80 08 00 00           	je	2176 <_main+0x9e0>
100002fb0: 8b 44 24 1c                 	movl	28(%rsp), %eax
100002fb4: 83 f8 03                    	cmpl	$3, %eax
100002fb7: 0f 8d 53 03 00 00           	jge	851 <_main+0x4c0>
100002fbd: 48 63 4c 24 20              	movslq	32(%rsp), %rcx
100002fc2: 48 63 74 24 24              	movslq	36(%rsp), %rsi
100002fc7: 48 0f af f1                 	imulq	%rcx, %rsi
100002fcb: 85 c0                       	testl	%eax, %eax
100002fcd: 0f 84 5d 08 00 00           	je	2141 <_main+0x9e0>
100002fd3: 48 85 f6                    	testq	%rsi, %rsi
100002fd6: 0f 84 54 08 00 00           	je	2132 <_main+0x9e0>
100002fdc: bf 19 00 00 00              	movl	$25, %edi
100002fe1: c5 f8 77                    	vzeroupper
100002fe4: e8 41 3e 00 00              	callq	15937 <dyld_stub_binder+0x100006e2a>
100002fe9: 3c 1b                       	cmpb	$27, %al
100002feb: 0f 84 3f 08 00 00           	je	2111 <_main+0x9e0>
100002ff1: e8 7c 3e 00 00              	callq	15996 <dyld_stub_binder+0x100006e72>
100002ff6: 49 89 c6                    	movq	%rax, %r14
100002ff9: 48 8d 9c 24 00 01 00 00     	leaq	256(%rsp), %rbx
100003001: 48 89 df                    	movq	%rbx, %rdi
100003004: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100003009: 48 8d 94 24 08 02 00 00     	leaq	520(%rsp), %rdx
100003011: c5 f9 6e 05 73 40 00 00     	vmovd	16499(%rip), %xmm0
100003019: e8 82 f6 ff ff              	callq	-2430 <__Z14get_predictionRN2cv3MatER14ModelInterfacef>
10000301e: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100003026: c5 fa 7e 05 22 40 00 00     	vmovq	16418(%rip), %xmm0
10000302e: 48 89 de                    	movq	%rbx, %rsi
100003031: e8 06 3e 00 00              	callq	15878 <dyld_stub_binder+0x100006e3c>
100003036: 48 8b 84 24 38 01 00 00     	movq	312(%rsp), %rax
10000303e: 48 85 c0                    	testq	%rax, %rax
100003041: 74 0e                       	je	14 <_main+0x201>
100003043: f0                          	lock
100003044: ff 48 14                    	decl	20(%rax)
100003047: 75 08                       	jne	8 <_main+0x201>
100003049: 48 89 df                    	movq	%rbx, %rdi
10000304c: e8 af 3d 00 00              	callq	15791 <dyld_stub_binder+0x100006e00>
100003051: 48 c7 84 24 38 01 00 00 00 00 00 00 	movq	$0, 312(%rsp)
10000305d: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
100003065: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003069: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
10000306d: 83 bc 24 04 01 00 00 00     	cmpl	$0, 260(%rsp)
100003075: 7e 30                       	jle	48 <_main+0x257>
100003077: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
10000307f: 31 c9                       	xorl	%ecx, %ecx
100003081: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000308b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003090: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003097: 48 ff c1                    	incq	%rcx
10000309a: 48 63 94 24 04 01 00 00     	movslq	260(%rsp), %rdx
1000030a2: 48 39 d1                    	cmpq	%rdx, %rcx
1000030a5: 7c e9                       	jl	-23 <_main+0x240>
1000030a7: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
1000030af: 48 8d 84 24 50 01 00 00     	leaq	336(%rsp), %rax
1000030b7: 48 39 c7                    	cmpq	%rax, %rdi
1000030ba: 74 08                       	je	8 <_main+0x274>
1000030bc: c5 f8 77                    	vzeroupper
1000030bf: e8 72 3d 00 00              	callq	15730 <dyld_stub_binder+0x100006e36>
1000030c4: c5 f8 77                    	vzeroupper
1000030c7: e8 a6 3d 00 00              	callq	15782 <dyld_stub_binder+0x100006e72>
1000030cc: 49 89 c4                    	movq	%rax, %r12
1000030cf: c7 84 24 00 01 00 00 00 00 ff 42    	movl	$1124007936, 256(%rsp)
1000030da: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
1000030e2: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000030e6: c5 fe 7f 40 f4              	vmovdqu	%ymm0, -12(%rax)
1000030eb: c5 fe 7f 40 10              	vmovdqu	%ymm0, 16(%rax)
1000030f0: 48 8b 44 24 20              	movq	32(%rsp), %rax
1000030f5: 48 8d 8c 24 08 01 00 00     	leaq	264(%rsp), %rcx
1000030fd: 48 89 8c 24 40 01 00 00     	movq	%rcx, 320(%rsp)
100003105: 48 8d 8c 24 50 01 00 00     	leaq	336(%rsp), %rcx
10000310d: 48 89 8c 24 48 01 00 00     	movq	%rcx, 328(%rsp)
100003115: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003119: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
10000311d: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
100003125: 48 89 df                    	movq	%rbx, %rdi
100003128: be 02 00 00 00              	movl	$2, %esi
10000312d: 4c 89 fa                    	movq	%r15, %rdx
100003130: 31 c9                       	xorl	%ecx, %ecx
100003132: c5 f8 77                    	vzeroupper
100003135: e8 cc 3c 00 00              	callq	15564 <dyld_stub_binder+0x100006e06>
10000313a: 48 c7 84 24 88 00 00 00 00 00 00 00 	movq	$0, 136(%rsp)
100003146: c7 44 24 78 00 00 06 c1     	movl	$3238395904, 120(%rsp)
10000314e: 48 8d 84 24 60 02 00 00     	leaq	608(%rsp), %rax
100003156: 48 89 84 24 80 00 00 00     	movq	%rax, 128(%rsp)
10000315e: 48 c7 84 24 70 01 00 00 00 00 00 00 	movq	$0, 368(%rsp)
10000316a: c7 84 24 60 01 00 00 00 00 01 02    	movl	$33619968, 352(%rsp)
100003175: 48 89 9c 24 68 01 00 00     	movq	%rbx, 360(%rsp)
10000317d: 8b 44 24 20                 	movl	32(%rsp), %eax
100003181: 8b 4c 24 24                 	movl	36(%rsp), %ecx
100003185: 89 8c 24 b0 01 00 00        	movl	%ecx, 432(%rsp)
10000318c: 89 84 24 b4 01 00 00        	movl	%eax, 436(%rsp)
100003193: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003197: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
10000319b: 48 8d 5c 24 78              	leaq	120(%rsp), %rbx
1000031a0: 48 89 df                    	movq	%rbx, %rdi
1000031a3: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
1000031ab: 48 8d 94 24 b0 01 00 00     	leaq	432(%rsp), %rdx
1000031b3: b9 01 00 00 00              	movl	$1, %ecx
1000031b8: e8 61 3c 00 00              	callq	15457 <dyld_stub_binder+0x100006e1e>
1000031bd: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000031c1: c5 fd 7f 84 24 60 01 00 00  	vmovdqa	%ymm0, 352(%rsp)
1000031ca: c7 44 24 78 00 00 ff 42     	movl	$1124007936, 120(%rsp)
1000031d2: 48 8d 44 24 7c              	leaq	124(%rsp), %rax
1000031d7: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
1000031dc: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
1000031e0: 48 8b 44 24 20              	movq	32(%rsp), %rax
1000031e5: 48 8d 8c 24 80 00 00 00     	leaq	128(%rsp), %rcx
1000031ed: 48 89 8c 24 b8 00 00 00     	movq	%rcx, 184(%rsp)
1000031f5: 48 8d 8c 24 c8 00 00 00     	leaq	200(%rsp), %rcx
1000031fd: 48 89 8c 24 c0 00 00 00     	movq	%rcx, 192(%rsp)
100003205: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003209: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
10000320d: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
100003215: 48 89 df                    	movq	%rbx, %rdi
100003218: be 02 00 00 00              	movl	$2, %esi
10000321d: 4c 89 fa                    	movq	%r15, %rdx
100003220: b9 10 00 00 00              	movl	$16, %ecx
100003225: c5 f8 77                    	vzeroupper
100003228: e8 d9 3b 00 00              	callq	15321 <dyld_stub_binder+0x100006e06>
10000322d: 48 89 df                    	movq	%rbx, %rdi
100003230: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003238: e8 d5 3b 00 00              	callq	15317 <dyld_stub_binder+0x100006e12>
10000323d: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003242: 48 85 c0                    	testq	%rax, %rax
100003245: 74 04                       	je	4 <_main+0x3fb>
100003247: f0                          	lock
100003248: ff 40 14                    	incl	20(%rax)
10000324b: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
100003253: 48 85 c0                    	testq	%rax, %rax
100003256: 74 10                       	je	16 <_main+0x418>
100003258: f0                          	lock
100003259: ff 48 14                    	decl	20(%rax)
10000325c: 75 0a                       	jne	10 <_main+0x418>
10000325e: 48 8d 7c 24 78              	leaq	120(%rsp), %rdi
100003263: e8 98 3b 00 00              	callq	15256 <dyld_stub_binder+0x100006e00>
100003268: 48 c7 84 24 b0 00 00 00 00 00 00 00 	movq	$0, 176(%rsp)
100003274: 48 8d 44 24 7c              	leaq	124(%rsp), %rax
100003279: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000327d: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100003282: 83 7c 24 7c 00              	cmpl	$0, 124(%rsp)
100003287: 0f 8e 22 06 00 00           	jle	1570 <_main+0xa5f>
10000328d: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003295: 31 c9                       	xorl	%ecx, %ecx
100003297: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
1000032a0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000032a7: 48 ff c1                    	incq	%rcx
1000032aa: 48 63 54 24 7c              	movslq	124(%rsp), %rdx
1000032af: 48 39 d1                    	cmpq	%rdx, %rcx
1000032b2: 7c ec                       	jl	-20 <_main+0x450>
1000032b4: 8b 44 24 18                 	movl	24(%rsp), %eax
1000032b8: 89 44 24 78                 	movl	%eax, 120(%rsp)
1000032bc: 83 fa 02                    	cmpl	$2, %edx
1000032bf: 0f 8f ff 05 00 00           	jg	1535 <_main+0xa74>
1000032c5: 8b 44 24 1c                 	movl	28(%rsp), %eax
1000032c9: 83 f8 02                    	cmpl	$2, %eax
1000032cc: 0f 8f f2 05 00 00           	jg	1522 <_main+0xa74>
1000032d2: 89 44 24 7c                 	movl	%eax, 124(%rsp)
1000032d6: 8b 4c 24 20                 	movl	32(%rsp), %ecx
1000032da: 8b 44 24 24                 	movl	36(%rsp), %eax
1000032de: 89 8c 24 80 00 00 00        	movl	%ecx, 128(%rsp)
1000032e5: 89 84 24 84 00 00 00        	movl	%eax, 132(%rsp)
1000032ec: 48 8b 44 24 60              	movq	96(%rsp), %rax
1000032f1: 48 8b 10                    	movq	(%rax), %rdx
1000032f4: 48 8b b4 24 c0 00 00 00     	movq	192(%rsp), %rsi
1000032fc: 48 89 16                    	movq	%rdx, (%rsi)
1000032ff: 48 8b 40 08                 	movq	8(%rax), %rax
100003303: 48 89 46 08                 	movq	%rax, 8(%rsi)
100003307: e9 ce 05 00 00              	jmp	1486 <_main+0xa8a>
10000330c: 0f 1f 40 00                 	nopl	(%rax)
100003310: 48 8b 4c 24 58              	movq	88(%rsp), %rcx
100003315: 83 f8 0f                    	cmpl	$15, %eax
100003318: 77 0c                       	ja	12 <_main+0x4d6>
10000331a: be 01 00 00 00              	movl	$1, %esi
10000331f: 31 d2                       	xorl	%edx, %edx
100003321: e9 ea 04 00 00              	jmp	1258 <_main+0x9c0>
100003326: 89 c2                       	movl	%eax, %edx
100003328: 83 e2 f0                    	andl	$-16, %edx
10000332b: 48 8d 72 f0                 	leaq	-16(%rdx), %rsi
10000332f: 48 89 f7                    	movq	%rsi, %rdi
100003332: 48 c1 ef 04                 	shrq	$4, %rdi
100003336: 48 ff c7                    	incq	%rdi
100003339: 89 fb                       	movl	%edi, %ebx
10000333b: 83 e3 03                    	andl	$3, %ebx
10000333e: 48 83 fe 30                 	cmpq	$48, %rsi
100003342: 73 25                       	jae	37 <_main+0x519>
100003344: c4 e2 7d 59 05 fb 3c 00 00  	vpbroadcastq	15611(%rip), %ymm0
10000334d: 31 ff                       	xorl	%edi, %edi
10000334f: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
100003353: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100003357: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
10000335b: 48 85 db                    	testq	%rbx, %rbx
10000335e: 0f 85 0e 03 00 00           	jne	782 <_main+0x822>
100003364: e9 d0 03 00 00              	jmp	976 <_main+0x8e9>
100003369: 48 89 de                    	movq	%rbx, %rsi
10000336c: 48 29 fe                    	subq	%rdi, %rsi
10000336f: c4 e2 7d 59 05 d0 3c 00 00  	vpbroadcastq	15568(%rip), %ymm0
100003378: 31 ff                       	xorl	%edi, %edi
10000337a: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
10000337e: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100003382: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
100003386: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003390: c4 e2 7d 25 24 b9           	vpmovsxdq	(%rcx,%rdi,4), %ymm4
100003396: c4 e2 7d 25 6c b9 10        	vpmovsxdq	16(%rcx,%rdi,4), %ymm5
10000339d: c4 e2 7d 25 74 b9 20        	vpmovsxdq	32(%rcx,%rdi,4), %ymm6
1000033a4: c4 e2 7d 25 7c b9 30        	vpmovsxdq	48(%rcx,%rdi,4), %ymm7
1000033ab: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
1000033b0: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
1000033b4: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
1000033b9: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
1000033be: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
1000033c3: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000033c9: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000033cd: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000033d2: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
1000033d7: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
1000033db: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
1000033e0: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
1000033e5: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
1000033e9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000033ee: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000033f2: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000033f6: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
1000033fb: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
1000033ff: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100003404: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100003408: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000340c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003411: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003415: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003419: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
10000341e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
100003422: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
100003427: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
10000342b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000342f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003434: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003438: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000343c: c4 e2 7d 25 64 b9 40        	vpmovsxdq	64(%rcx,%rdi,4), %ymm4
100003443: c4 e2 7d 25 6c b9 50        	vpmovsxdq	80(%rcx,%rdi,4), %ymm5
10000344a: c4 e2 7d 25 74 b9 60        	vpmovsxdq	96(%rcx,%rdi,4), %ymm6
100003451: c4 e2 7d 25 7c b9 70        	vpmovsxdq	112(%rcx,%rdi,4), %ymm7
100003458: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
10000345d: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100003462: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100003467: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
10000346b: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100003470: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003476: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000347a: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
10000347f: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100003484: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100003488: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
10000348d: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100003491: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100003496: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000349b: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
10000349f: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000034a3: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
1000034a8: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
1000034ac: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
1000034b1: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
1000034b5: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000034b9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000034be: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
1000034c2: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000034c6: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
1000034cb: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
1000034cf: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
1000034d4: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
1000034d8: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000034dc: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000034e1: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
1000034e5: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000034e9: c4 e2 7d 25 a4 b9 80 00 00 00       	vpmovsxdq	128(%rcx,%rdi,4), %ymm4
1000034f3: c4 e2 7d 25 ac b9 90 00 00 00       	vpmovsxdq	144(%rcx,%rdi,4), %ymm5
1000034fd: c4 e2 7d 25 b4 b9 a0 00 00 00       	vpmovsxdq	160(%rcx,%rdi,4), %ymm6
100003507: c4 e2 7d 25 bc b9 b0 00 00 00       	vpmovsxdq	176(%rcx,%rdi,4), %ymm7
100003511: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
100003516: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
10000351b: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100003520: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
100003524: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100003529: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
10000352f: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100003533: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003538: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
10000353d: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100003541: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
100003546: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
10000354a: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
10000354f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003554: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003558: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
10000355c: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100003561: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100003565: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
10000356a: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
10000356e: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100003572: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003577: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
10000357b: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
10000357f: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
100003584: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100003588: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
10000358d: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100003591: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100003595: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000359a: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
10000359e: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000035a2: c4 e2 7d 25 a4 b9 c0 00 00 00       	vpmovsxdq	192(%rcx,%rdi,4), %ymm4
1000035ac: c4 e2 7d 25 ac b9 d0 00 00 00       	vpmovsxdq	208(%rcx,%rdi,4), %ymm5
1000035b6: c4 e2 7d 25 b4 b9 e0 00 00 00       	vpmovsxdq	224(%rcx,%rdi,4), %ymm6
1000035c0: c4 e2 7d 25 bc b9 f0 00 00 00       	vpmovsxdq	240(%rcx,%rdi,4), %ymm7
1000035ca: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
1000035cf: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
1000035d4: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
1000035d9: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
1000035dd: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
1000035e2: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000035e8: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000035ec: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000035f1: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
1000035f6: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
1000035fa: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
1000035ff: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100003603: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100003608: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000360d: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003611: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100003615: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
10000361a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
10000361e: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100003623: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100003627: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
10000362b: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003630: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003634: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003638: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
10000363d: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100003641: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100003646: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
10000364a: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
10000364e: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003653: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003657: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000365b: 48 83 c7 40                 	addq	$64, %rdi
10000365f: 48 83 c6 04                 	addq	$4, %rsi
100003663: 0f 85 27 fd ff ff           	jne	-729 <_main+0x540>
100003669: 48 85 db                    	testq	%rbx, %rbx
10000366c: 0f 84 c7 00 00 00           	je	199 <_main+0x8e9>
100003672: 48 8d 34 b9                 	leaq	(%rcx,%rdi,4), %rsi
100003676: 48 83 c6 30                 	addq	$48, %rsi
10000367a: 48 c1 e3 06                 	shlq	$6, %rbx
10000367e: 31 ff                       	xorl	%edi, %edi
100003680: c4 e2 7d 25 64 3e d0        	vpmovsxdq	-48(%rsi,%rdi), %ymm4
100003687: c4 e2 7d 25 6c 3e e0        	vpmovsxdq	-32(%rsi,%rdi), %ymm5
10000368e: c4 e2 7d 25 74 3e f0        	vpmovsxdq	-16(%rsi,%rdi), %ymm6
100003695: c4 e2 7d 25 3c 3e           	vpmovsxdq	(%rsi,%rdi), %ymm7
10000369b: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
1000036a0: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
1000036a4: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
1000036a9: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
1000036ae: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
1000036b3: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000036b9: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000036bd: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000036c2: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
1000036c7: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
1000036cb: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
1000036d0: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
1000036d5: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
1000036d9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000036de: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000036e2: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000036e6: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
1000036eb: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
1000036ef: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
1000036f4: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
1000036f8: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
1000036fc: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003701: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003705: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003709: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
10000370e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
100003712: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
100003717: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
10000371b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000371f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003724: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003728: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000372c: 48 83 c7 40                 	addq	$64, %rdi
100003730: 48 39 fb                    	cmpq	%rdi, %rbx
100003733: 0f 85 47 ff ff ff           	jne	-185 <_main+0x830>
100003739: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
10000373e: c5 dd f4 e0                 	vpmuludq	%ymm0, %ymm4, %ymm4
100003742: c5 d5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm5
100003747: c5 e5 f4 ed                 	vpmuludq	%ymm5, %ymm3, %ymm5
10000374b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000374f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003754: c5 e5 f4 c0                 	vpmuludq	%ymm0, %ymm3, %ymm0
100003758: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
10000375c: c5 e5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm3
100003761: c5 e5 f4 d8                 	vpmuludq	%ymm0, %ymm3, %ymm3
100003765: c5 dd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm4
10000376a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
10000376e: c5 dd d4 db                 	vpaddq	%ymm3, %ymm4, %ymm3
100003772: c5 e5 73 f3 20              	vpsllq	$32, %ymm3, %ymm3
100003777: c5 ed f4 c0                 	vpmuludq	%ymm0, %ymm2, %ymm0
10000377b: c5 fd d4 c3                 	vpaddq	%ymm3, %ymm0, %ymm0
10000377f: c5 ed 73 d1 20              	vpsrlq	$32, %ymm1, %ymm2
100003784: c5 ed f4 d0                 	vpmuludq	%ymm0, %ymm2, %ymm2
100003788: c5 e5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm3
10000378d: c5 f5 f4 db                 	vpmuludq	%ymm3, %ymm1, %ymm3
100003791: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
100003795: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
10000379a: c5 f5 f4 c0                 	vpmuludq	%ymm0, %ymm1, %ymm0
10000379e: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
1000037a2: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
1000037a8: c5 ed 73 d0 20              	vpsrlq	$32, %ymm0, %ymm2
1000037ad: c5 ed f4 d1                 	vpmuludq	%ymm1, %ymm2, %ymm2
1000037b1: c5 e5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm3
1000037b6: c5 fd f4 db                 	vpmuludq	%ymm3, %ymm0, %ymm3
1000037ba: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
1000037be: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
1000037c3: c5 fd f4 c1                 	vpmuludq	%ymm1, %ymm0, %ymm0
1000037c7: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
1000037cb: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
1000037d0: c5 e9 73 d0 20              	vpsrlq	$32, %xmm0, %xmm2
1000037d5: c5 e9 f4 d1                 	vpmuludq	%xmm1, %xmm2, %xmm2
1000037d9: c5 e1 73 d8 0c              	vpsrldq	$12, %xmm0, %xmm3
1000037de: c5 f9 f4 db                 	vpmuludq	%xmm3, %xmm0, %xmm3
1000037e2: c5 e1 d4 d2                 	vpaddq	%xmm2, %xmm3, %xmm2
1000037e6: c5 e9 73 f2 20              	vpsllq	$32, %xmm2, %xmm2
1000037eb: c5 f9 f4 c1                 	vpmuludq	%xmm1, %xmm0, %xmm0
1000037ef: c5 f9 d4 c2                 	vpaddq	%xmm2, %xmm0, %xmm0
1000037f3: c4 e1 f9 7e c6              	vmovq	%xmm0, %rsi
1000037f8: 48 39 c2                    	cmpq	%rax, %rdx
1000037fb: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100003803: 74 1b                       	je	27 <_main+0x9d0>
100003805: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000380f: 90                          	nop
100003810: 48 63 3c 91                 	movslq	(%rcx,%rdx,4), %rdi
100003814: 48 0f af f7                 	imulq	%rdi, %rsi
100003818: 48 ff c2                    	incq	%rdx
10000381b: 48 39 d0                    	cmpq	%rdx, %rax
10000381e: 75 f0                       	jne	-16 <_main+0x9c0>
100003820: 85 c0                       	testl	%eax, %eax
100003822: 0f 85 ab f7 ff ff           	jne	-2133 <_main+0x183>
100003828: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100003830: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003835: 48 85 c0                    	testq	%rax, %rax
100003838: 74 13                       	je	19 <_main+0x9fd>
10000383a: f0                          	lock
10000383b: ff 48 14                    	decl	20(%rax)
10000383e: 75 0d                       	jne	13 <_main+0x9fd>
100003840: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
100003845: c5 f8 77                    	vzeroupper
100003848: e8 b3 35 00 00              	callq	13747 <dyld_stub_binder+0x100006e00>
10000384d: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
100003856: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000385a: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
10000385f: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100003864: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
100003869: 7e 29                       	jle	41 <_main+0xa44>
10000386b: 48 8b 44 24 58              	movq	88(%rsp), %rax
100003870: 31 c9                       	xorl	%ecx, %ecx
100003872: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000387c: 0f 1f 40 00                 	nopl	(%rax)
100003880: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003887: 48 ff c1                    	incq	%rcx
10000388a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000388f: 48 39 d1                    	cmpq	%rdx, %rcx
100003892: 7c ec                       	jl	-20 <_main+0xa30>
100003894: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003899: 4c 39 f7                    	cmpq	%r14, %rdi
10000389c: 0f 84 9e f6 ff ff           	je	-2402 <_main+0xf0>
1000038a2: c5 f8 77                    	vzeroupper
1000038a5: e8 8c 35 00 00              	callq	13708 <dyld_stub_binder+0x100006e36>
1000038aa: e9 91 f6 ff ff              	jmp	-2415 <_main+0xf0>
1000038af: 8b 44 24 18                 	movl	24(%rsp), %eax
1000038b3: 89 44 24 78                 	movl	%eax, 120(%rsp)
1000038b7: 8b 44 24 1c                 	movl	28(%rsp), %eax
1000038bb: 83 f8 02                    	cmpl	$2, %eax
1000038be: 0f 8e 0e fa ff ff           	jle	-1522 <_main+0x482>
1000038c4: 48 8d 7c 24 78              	leaq	120(%rsp), %rdi
1000038c9: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
1000038ce: c5 f8 77                    	vzeroupper
1000038d1: e8 36 35 00 00              	callq	13622 <dyld_stub_binder+0x100006e0c>
1000038d6: 8b 4c 24 20                 	movl	32(%rsp), %ecx
1000038da: c4 c1 eb 2a c6              	vcvtsi2sd	%r14, %xmm2, %xmm0
1000038df: c4 c1 eb 2a cc              	vcvtsi2sd	%r12, %xmm2, %xmm1
1000038e4: c5 fb 10 15 54 37 00 00     	vmovsd	14164(%rip), %xmm2
1000038ec: c5 fb 5e c2                 	vdivsd	%xmm2, %xmm0, %xmm0
1000038f0: c5 f3 5e ca                 	vdivsd	%xmm2, %xmm1, %xmm1
1000038f4: c5 fc 10 54 24 28           	vmovups	40(%rsp), %ymm2
1000038fa: c5 fc 11 94 24 88 00 00 00  	vmovups	%ymm2, 136(%rsp)
100003903: c5 f9 10 54 24 48           	vmovupd	72(%rsp), %xmm2
100003909: c5 f9 11 94 24 a8 00 00 00  	vmovupd	%xmm2, 168(%rsp)
100003912: 85 c9                       	testl	%ecx, %ecx
100003914: 4d 89 fe                    	movq	%r15, %r14
100003917: 0f 84 49 01 00 00           	je	329 <_main+0xc16>
10000391d: 31 c0                       	xorl	%eax, %eax
10000391f: 8b 74 24 24                 	movl	36(%rsp), %esi
100003923: 85 f6                       	testl	%esi, %esi
100003925: be 00 00 00 00              	movl	$0, %esi
10000392a: 75 17                       	jne	23 <_main+0xaf3>
10000392c: 0f 1f 40 00                 	nopl	(%rax)
100003930: ff c0                       	incl	%eax
100003932: 39 c8                       	cmpl	%ecx, %eax
100003934: 0f 83 2c 01 00 00           	jae	300 <_main+0xc16>
10000393a: 85 f6                       	testl	%esi, %esi
10000393c: be 00 00 00 00              	movl	$0, %esi
100003941: 74 ed                       	je	-19 <_main+0xae0>
100003943: 48 63 c8                    	movslq	%eax, %rcx
100003946: 31 d2                       	xorl	%edx, %edx
100003948: c5 fb 10 25 08 37 00 00     	vmovsd	14088(%rip), %xmm4
100003950: c5 fa 10 2d 38 37 00 00     	vmovss	14136(%rip), %xmm5
100003958: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100003960: 48 8b 74 24 60              	movq	96(%rsp), %rsi
100003965: 48 8b 3e                    	movq	(%rsi), %rdi
100003968: 48 0f af f9                 	imulq	%rcx, %rdi
10000396c: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100003971: 48 63 d2                    	movslq	%edx, %rdx
100003974: 48 8d 34 52                 	leaq	(%rdx,%rdx,2), %rsi
100003978: 0f b6 3c 37                 	movzbl	(%rdi,%rsi), %edi
10000397c: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003980: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003984: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003988: 48 8b 9c 24 c0 00 00 00     	movq	192(%rsp), %rbx
100003990: 48 8b 1b                    	movq	(%rbx), %rbx
100003993: 48 0f af d9                 	imulq	%rcx, %rbx
100003997: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
10000399f: 40 88 3c 33                 	movb	%dil, (%rbx,%rsi)
1000039a3: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
1000039a8: 48 8b 3f                    	movq	(%rdi), %rdi
1000039ab: 48 0f af f9                 	imulq	%rcx, %rdi
1000039af: 48 03 7c 24 28              	addq	40(%rsp), %rdi
1000039b4: 0f b6 7c 37 01              	movzbl	1(%rdi,%rsi), %edi
1000039b9: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
1000039bd: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
1000039c5: 48 8b 3f                    	movq	(%rdi), %rdi
1000039c8: 48 0f af f9                 	imulq	%rcx, %rdi
1000039cc: 48 03 bc 24 10 01 00 00     	addq	272(%rsp), %rdi
1000039d4: 0f b6 3c 3a                 	movzbl	(%rdx,%rdi), %edi
1000039d8: c5 ca 2a df                 	vcvtsi2ss	%edi, %xmm6, %xmm3
1000039dc: c5 e2 59 dd                 	vmulss	%xmm5, %xmm3, %xmm3
1000039e0: c5 e2 5a db                 	vcvtss2sd	%xmm3, %xmm3, %xmm3
1000039e4: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
1000039e8: c5 eb 58 d3                 	vaddsd	%xmm3, %xmm2, %xmm2
1000039ec: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
1000039f0: 48 8b 9c 24 c0 00 00 00     	movq	192(%rsp), %rbx
1000039f8: 48 8b 1b                    	movq	(%rbx), %rbx
1000039fb: 48 0f af d9                 	imulq	%rcx, %rbx
1000039ff: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100003a07: 40 88 7c 33 01              	movb	%dil, 1(%rbx,%rsi)
100003a0c: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003a11: 48 8b 3f                    	movq	(%rdi), %rdi
100003a14: 48 0f af f9                 	imulq	%rcx, %rdi
100003a18: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100003a1d: 0f b6 7c 37 02              	movzbl	2(%rdi,%rsi), %edi
100003a22: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003a26: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003a2a: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003a2e: 48 8b 9c 24 c0 00 00 00     	movq	192(%rsp), %rbx
100003a36: 48 8b 1b                    	movq	(%rbx), %rbx
100003a39: 48 0f af d9                 	imulq	%rcx, %rbx
100003a3d: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100003a45: 40 88 7c 33 02              	movb	%dil, 2(%rbx,%rsi)
100003a4a: ff c2                       	incl	%edx
100003a4c: 8b 74 24 24                 	movl	36(%rsp), %esi
100003a50: 39 f2                       	cmpl	%esi, %edx
100003a52: 0f 82 08 ff ff ff           	jb	-248 <_main+0xb10>
100003a58: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100003a5c: ff c0                       	incl	%eax
100003a5e: 39 c8                       	cmpl	%ecx, %eax
100003a60: 0f 82 d4 fe ff ff           	jb	-300 <_main+0xaea>
100003a66: c5 fb 10 15 f2 35 00 00     	vmovsd	13810(%rip), %xmm2
100003a6e: c5 eb 59 94 24 d8 00 00 00  	vmulsd	216(%rsp), %xmm2, %xmm2
100003a77: c5 f3 5c c0                 	vsubsd	%xmm0, %xmm1, %xmm0
100003a7b: c5 fb 58 05 e5 35 00 00     	vaddsd	13797(%rip), %xmm0, %xmm0
100003a83: c5 fb 10 0d e5 35 00 00     	vmovsd	13797(%rip), %xmm1
100003a8b: c5 f3 5e c0                 	vdivsd	%xmm0, %xmm1, %xmm0
100003a8f: c5 eb 58 c0                 	vaddsd	%xmm0, %xmm2, %xmm0
100003a93: 8b 9c 24 28 02 00 00        	movl	552(%rsp), %ebx
100003a9a: c5 fb 11 84 24 d8 00 00 00  	vmovsd	%xmm0, 216(%rsp)
100003aa3: c5 f8 77                    	vzeroupper
100003aa6: e8 1b 34 00 00              	callq	13339 <dyld_stub_binder+0x100006ec6>
100003aab: c5 fb 2c f0                 	vcvttsd2si	%xmm0, %esi
100003aaf: 4c 89 ef                    	movq	%r13, %rdi
100003ab2: e8 d9 33 00 00              	callq	13273 <dyld_stub_binder+0x100006e90>
100003ab7: 4c 89 ef                    	movq	%r13, %rdi
100003aba: 31 f6                       	xorl	%esi, %esi
100003abc: 48 8d 15 1a 54 00 00        	leaq	21530(%rip), %rdx
100003ac3: e8 98 33 00 00              	callq	13208 <dyld_stub_binder+0x100006e60>
100003ac8: 48 8b 48 10                 	movq	16(%rax), %rcx
100003acc: 48 89 8c 24 f0 00 00 00     	movq	%rcx, 240(%rsp)
100003ad4: c5 f9 10 00                 	vmovupd	(%rax), %xmm0
100003ad8: c5 f9 29 84 24 e0 00 00 00  	vmovapd	%xmm0, 224(%rsp)
100003ae1: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003ae5: c5 f9 11 00                 	vmovupd	%xmm0, (%rax)
100003ae9: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003af1: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003af9: 48 8d 35 e4 53 00 00        	leaq	21476(%rip), %rsi
100003b00: e8 4f 33 00 00              	callq	13135 <dyld_stub_binder+0x100006e54>
100003b05: c4 e1 cb 2a c3              	vcvtsi2sd	%rbx, %xmm6, %xmm0
100003b0a: c5 fb 59 84 24 d8 00 00 00  	vmulsd	216(%rsp), %xmm0, %xmm0
100003b13: c5 fb 5e 05 5d 35 00 00     	vdivsd	13661(%rip), %xmm0, %xmm0
100003b1b: 48 8b 48 10                 	movq	16(%rax), %rcx
100003b1f: 48 89 8c 24 d0 03 00 00     	movq	%rcx, 976(%rsp)
100003b27: c5 f9 10 08                 	vmovupd	(%rax), %xmm1
100003b2b: c5 f9 29 8c 24 c0 03 00 00  	vmovapd	%xmm1, 960(%rsp)
100003b34: c5 f1 57 c9                 	vxorpd	%xmm1, %xmm1, %xmm1
100003b38: c5 f9 11 08                 	vmovupd	%xmm1, (%rax)
100003b3c: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003b44: 48 8d bc 24 98 01 00 00     	leaq	408(%rsp), %rdi
100003b4c: e8 39 33 00 00              	callq	13113 <dyld_stub_binder+0x100006e8a>
100003b51: 0f b6 94 24 98 01 00 00     	movzbl	408(%rsp), %edx
100003b59: f6 c2 01                    	testb	$1, %dl
100003b5c: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100003b64: 74 12                       	je	18 <_main+0xd28>
100003b66: 48 8b b4 24 a8 01 00 00     	movq	424(%rsp), %rsi
100003b6e: 48 8b 94 24 a0 01 00 00     	movq	416(%rsp), %rdx
100003b76: eb 0b                       	jmp	11 <_main+0xd33>
100003b78: 48 d1 ea                    	shrq	%rdx
100003b7b: 48 8d b4 24 99 01 00 00     	leaq	409(%rsp), %rsi
100003b83: 4c 89 f7                    	movq	%r14, %rdi
100003b86: e8 cf 32 00 00              	callq	13007 <dyld_stub_binder+0x100006e5a>
100003b8b: 48 8b 48 10                 	movq	16(%rax), %rcx
100003b8f: 48 89 8c 24 70 01 00 00     	movq	%rcx, 368(%rsp)
100003b97: c5 f8 10 00                 	vmovups	(%rax), %xmm0
100003b9b: c5 f8 29 84 24 60 01 00 00  	vmovaps	%xmm0, 352(%rsp)
100003ba4: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003ba8: c5 f8 11 00                 	vmovups	%xmm0, (%rax)
100003bac: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003bb4: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
100003bbc: 0f 85 7a 01 00 00           	jne	378 <_main+0xeec>
100003bc2: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003bca: 0f 85 87 01 00 00           	jne	391 <_main+0xf07>
100003bd0: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
100003bd8: 0f 85 94 01 00 00           	jne	404 <_main+0xf22>
100003bde: 4d 89 ec                    	movq	%r13, %r12
100003be1: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003be9: 74 0d                       	je	13 <_main+0xda8>
100003beb: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
100003bf3: e8 a4 32 00 00              	callq	12964 <dyld_stub_binder+0x100006e9c>
100003bf8: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
100003c04: c7 84 24 c0 03 00 00 00 00 01 03    	movl	$50397184, 960(%rsp)
100003c0f: 4c 8d 6c 24 78              	leaq	120(%rsp), %r13
100003c14: 4c 89 ac 24 c8 03 00 00     	movq	%r13, 968(%rsp)
100003c1c: 48 b8 1e 00 00 00 1e 00 00 00       	movabsq	$128849018910, %rax
100003c26: 48 89 84 24 b8 01 00 00     	movq	%rax, 440(%rsp)
100003c2e: c5 fc 28 05 8a 34 00 00     	vmovaps	13450(%rip), %ymm0
100003c36: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100003c3f: c7 44 24 08 00 00 00 00     	movl	$0, 8(%rsp)
100003c47: c7 04 24 10 00 00 00        	movl	$16, (%rsp)
100003c4e: 4c 89 f7                    	movq	%r14, %rdi
100003c51: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003c59: 48 8d 94 24 b8 01 00 00     	leaq	440(%rsp), %rdx
100003c61: 31 c9                       	xorl	%ecx, %ecx
100003c63: c5 fb 10 05 15 34 00 00     	vmovsd	13333(%rip), %xmm0
100003c6b: 4c 8d 84 24 40 02 00 00     	leaq	576(%rsp), %r8
100003c73: 41 b9 02 00 00 00           	movl	$2, %r9d
100003c79: c5 f8 77                    	vzeroupper
100003c7c: e8 a3 31 00 00              	callq	12707 <dyld_stub_binder+0x100006e24>
100003c81: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003c85: c5 f9 29 84 24 c0 03 00 00  	vmovapd	%xmm0, 960(%rsp)
100003c8e: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
100003c9a: c6 84 24 c0 03 00 00 0a     	movb	$10, 960(%rsp)
100003ca2: 48 8d 84 24 c1 03 00 00     	leaq	961(%rsp), %rax
100003caa: c6 40 04 65                 	movb	$101, 4(%rax)
100003cae: c7 00 66 72 61 6d           	movl	$1835102822, (%rax)
100003cb4: c6 84 24 c6 03 00 00 00     	movb	$0, 966(%rsp)
100003cbc: 48 c7 84 24 f0 00 00 00 00 00 00 00 	movq	$0, 240(%rsp)
100003cc8: c7 84 24 e0 00 00 00 00 00 01 01    	movl	$16842752, 224(%rsp)
100003cd3: 4c 89 ac 24 e8 00 00 00     	movq	%r13, 232(%rsp)
100003cdb: 4c 89 f7                    	movq	%r14, %rdi
100003cde: 48 8d b4 24 e0 00 00 00     	leaq	224(%rsp), %rsi
100003ce6: e8 2d 31 00 00              	callq	12589 <dyld_stub_binder+0x100006e18>
100003ceb: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003cf3: 4d 89 e5                    	movq	%r12, %r13
100003cf6: 4c 8d 74 24 68              	leaq	104(%rsp), %r14
100003cfb: 0f 85 94 00 00 00           	jne	148 <_main+0xf45>
100003d01: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003d09: 4c 8d 64 24 78              	leaq	120(%rsp), %r12
100003d0e: 0f 85 a1 00 00 00           	jne	161 <_main+0xf65>
100003d14: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
100003d1c: 48 85 c0                    	testq	%rax, %rax
100003d1f: 0f 84 ae 00 00 00           	je	174 <_main+0xf83>
100003d25: f0                          	lock
100003d26: ff 48 14                    	decl	20(%rax)
100003d29: 0f 85 a4 00 00 00           	jne	164 <_main+0xf83>
100003d2f: 4c 89 e7                    	movq	%r12, %rdi
100003d32: e8 c9 30 00 00              	callq	12489 <dyld_stub_binder+0x100006e00>
100003d37: e9 97 00 00 00              	jmp	151 <_main+0xf83>
100003d3c: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100003d44: e8 53 31 00 00              	callq	12627 <dyld_stub_binder+0x100006e9c>
100003d49: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003d51: 0f 84 79 fe ff ff           	je	-391 <_main+0xd80>
100003d57: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003d5f: e8 38 31 00 00              	callq	12600 <dyld_stub_binder+0x100006e9c>
100003d64: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
100003d6c: 0f 84 6c fe ff ff           	je	-404 <_main+0xd8e>
100003d72: 48 8b bc 24 f0 00 00 00     	movq	240(%rsp), %rdi
100003d7a: e8 1d 31 00 00              	callq	12573 <dyld_stub_binder+0x100006e9c>
100003d7f: 4d 89 ec                    	movq	%r13, %r12
100003d82: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003d8a: 0f 85 5b fe ff ff           	jne	-421 <_main+0xd9b>
100003d90: e9 63 fe ff ff              	jmp	-413 <_main+0xda8>
100003d95: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003d9d: e8 fa 30 00 00              	callq	12538 <dyld_stub_binder+0x100006e9c>
100003da2: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003daa: 4c 8d 64 24 78              	leaq	120(%rsp), %r12
100003daf: 0f 84 5f ff ff ff           	je	-161 <_main+0xec4>
100003db5: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
100003dbd: e8 da 30 00 00              	callq	12506 <dyld_stub_binder+0x100006e9c>
100003dc2: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
100003dca: 48 85 c0                    	testq	%rax, %rax
100003dcd: 0f 85 52 ff ff ff           	jne	-174 <_main+0xed5>
100003dd3: 48 c7 84 24 b0 00 00 00 00 00 00 00 	movq	$0, 176(%rsp)
100003ddf: 48 8d 44 24 7c              	leaq	124(%rsp), %rax
100003de4: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003de8: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100003ded: 83 7c 24 7c 00              	cmpl	$0, 124(%rsp)
100003df2: 7e 20                       	jle	32 <_main+0xfc4>
100003df4: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003dfc: 31 c9                       	xorl	%ecx, %ecx
100003dfe: 66 90                       	nop
100003e00: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003e07: 48 ff c1                    	incq	%rcx
100003e0a: 48 63 54 24 7c              	movslq	124(%rsp), %rdx
100003e0f: 48 39 d1                    	cmpq	%rdx, %rcx
100003e12: 7c ec                       	jl	-20 <_main+0xfb0>
100003e14: 48 8b bc 24 c0 00 00 00     	movq	192(%rsp), %rdi
100003e1c: 48 8d 84 24 c8 00 00 00     	leaq	200(%rsp), %rax
100003e24: 48 39 c7                    	cmpq	%rax, %rdi
100003e27: 74 08                       	je	8 <_main+0xfe1>
100003e29: c5 f8 77                    	vzeroupper
100003e2c: e8 05 30 00 00              	callq	12293 <dyld_stub_binder+0x100006e36>
100003e31: 48 8b 84 24 38 01 00 00     	movq	312(%rsp), %rax
100003e39: 48 85 c0                    	testq	%rax, %rax
100003e3c: 74 16                       	je	22 <_main+0x1004>
100003e3e: f0                          	lock
100003e3f: ff 48 14                    	decl	20(%rax)
100003e42: 75 10                       	jne	16 <_main+0x1004>
100003e44: 48 8d bc 24 00 01 00 00     	leaq	256(%rsp), %rdi
100003e4c: c5 f8 77                    	vzeroupper
100003e4f: e8 ac 2f 00 00              	callq	12204 <dyld_stub_binder+0x100006e00>
100003e54: 48 c7 84 24 38 01 00 00 00 00 00 00 	movq	$0, 312(%rsp)
100003e60: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
100003e68: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003e6c: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
100003e70: 83 bc 24 04 01 00 00 00     	cmpl	$0, 260(%rsp)
100003e78: 7e 2d                       	jle	45 <_main+0x1057>
100003e7a: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
100003e82: 31 c9                       	xorl	%ecx, %ecx
100003e84: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003e8e: 66 90                       	nop
100003e90: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003e97: 48 ff c1                    	incq	%rcx
100003e9a: 48 63 94 24 04 01 00 00     	movslq	260(%rsp), %rdx
100003ea2: 48 39 d1                    	cmpq	%rdx, %rcx
100003ea5: 7c e9                       	jl	-23 <_main+0x1040>
100003ea7: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
100003eaf: 48 8d 84 24 50 01 00 00     	leaq	336(%rsp), %rax
100003eb7: 48 39 c7                    	cmpq	%rax, %rdi
100003eba: 74 08                       	je	8 <_main+0x1074>
100003ebc: c5 f8 77                    	vzeroupper
100003ebf: e8 72 2f 00 00              	callq	12146 <dyld_stub_binder+0x100006e36>
100003ec4: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100003ecc: c5 f8 77                    	vzeroupper
100003ecf: e8 9c 04 00 00              	callq	1180 <_main+0x1520>
100003ed4: 45 31 e4                    	xorl	%r12d, %r12d
100003ed7: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003edc: 48 85 c0                    	testq	%rax, %rax
100003edf: 0f 85 55 f9 ff ff           	jne	-1707 <_main+0x9ea>
100003ee5: e9 63 f9 ff ff              	jmp	-1693 <_main+0x9fd>
100003eea: 48 8b 3d 57 51 00 00        	movq	20823(%rip), %rdi
100003ef1: 48 8d 35 00 50 00 00        	leaq	20480(%rip), %rsi
100003ef8: ba 0d 00 00 00              	movl	$13, %edx
100003efd: c5 f8 77                    	vzeroupper
100003f00: e8 fb 05 00 00              	callq	1531 <_main+0x16b0>
100003f05: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100003f0d: e8 e2 2e 00 00              	callq	12002 <dyld_stub_binder+0x100006df4>
100003f12: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100003f1a: e8 11 0a 00 00              	callq	2577 <__ZN14ModelInterfaceD2Ev>
100003f1f: 48 8b 05 3a 51 00 00        	movq	20794(%rip), %rax
100003f26: 48 8b 00                    	movq	(%rax), %rax
100003f29: 48 3b 84 24 e0 03 00 00     	cmpq	992(%rsp), %rax
100003f31: 75 11                       	jne	17 <_main+0x10f4>
100003f33: 31 c0                       	xorl	%eax, %eax
100003f35: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100003f39: 5b                          	popq	%rbx
100003f3a: 41 5c                       	popq	%r12
100003f3c: 41 5d                       	popq	%r13
100003f3e: 41 5e                       	popq	%r14
100003f40: 41 5f                       	popq	%r15
100003f42: 5d                          	popq	%rbp
100003f43: c3                          	retq
100003f44: e8 71 2f 00 00              	callq	12145 <dyld_stub_binder+0x100006eba>
100003f49: e9 e7 03 00 00              	jmp	999 <_main+0x14e5>
100003f4e: 48 89 c3                    	movq	%rax, %rbx
100003f51: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100003f59: 0f 84 e9 03 00 00           	je	1001 <_main+0x14f8>
100003f5f: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
100003f67: e8 30 2f 00 00              	callq	12080 <dyld_stub_binder+0x100006e9c>
100003f6c: e9 d7 03 00 00              	jmp	983 <_main+0x14f8>
100003f71: 48 89 c3                    	movq	%rax, %rbx
100003f74: e9 cf 03 00 00              	jmp	975 <_main+0x14f8>
100003f79: 48 89 c7                    	movq	%rax, %rdi
100003f7c: e8 df 03 00 00              	callq	991 <_main+0x1510>
100003f81: 48 89 c7                    	movq	%rax, %rdi
100003f84: e8 d7 03 00 00              	callq	983 <_main+0x1510>
100003f89: 48 89 c7                    	movq	%rax, %rdi
100003f8c: e8 cf 03 00 00              	callq	975 <_main+0x1510>
100003f91: 48 89 c3                    	movq	%rax, %rbx
100003f94: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
100003f9c: 48 85 c0                    	testq	%rax, %rax
100003f9f: 0f 85 c8 01 00 00           	jne	456 <_main+0x131d>
100003fa5: e9 d3 01 00 00              	jmp	467 <_main+0x132d>
100003faa: 48 89 c3                    	movq	%rax, %rbx
100003fad: 48 8b 84 24 38 01 00 00     	movq	312(%rsp), %rax
100003fb5: 48 85 c0                    	testq	%rax, %rax
100003fb8: 74 13                       	je	19 <_main+0x117d>
100003fba: f0                          	lock
100003fbb: ff 48 14                    	decl	20(%rax)
100003fbe: 75 0d                       	jne	13 <_main+0x117d>
100003fc0: 48 8d bc 24 00 01 00 00     	leaq	256(%rsp), %rdi
100003fc8: e8 33 2e 00 00              	callq	11827 <dyld_stub_binder+0x100006e00>
100003fcd: 48 c7 84 24 38 01 00 00 00 00 00 00 	movq	$0, 312(%rsp)
100003fd9: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003fdd: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
100003fe5: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100003fe9: 83 bc 24 04 01 00 00 00     	cmpl	$0, 260(%rsp)
100003ff1: 7e 21                       	jle	33 <_main+0x11c4>
100003ff3: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
100003ffb: 31 c9                       	xorl	%ecx, %ecx
100003ffd: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004004: 48 ff c1                    	incq	%rcx
100004007: 48 63 94 24 04 01 00 00     	movslq	260(%rsp), %rdx
10000400f: 48 39 d1                    	cmpq	%rdx, %rcx
100004012: 7c e9                       	jl	-23 <_main+0x11ad>
100004014: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
10000401c: 48 8d 84 24 50 01 00 00     	leaq	336(%rsp), %rax
100004024: 48 39 c7                    	cmpq	%rax, %rdi
100004027: 0f 84 88 02 00 00           	je	648 <_main+0x1465>
10000402d: c5 f8 77                    	vzeroupper
100004030: e8 01 2e 00 00              	callq	11777 <dyld_stub_binder+0x100006e36>
100004035: e9 7b 02 00 00              	jmp	635 <_main+0x1465>
10000403a: 48 89 c7                    	movq	%rax, %rdi
10000403d: e8 1e 03 00 00              	callq	798 <_main+0x1510>
100004042: 48 89 c3                    	movq	%rax, %rbx
100004045: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000404a: 48 85 c0                    	testq	%rax, %rax
10000404d: 0f 85 6c 02 00 00           	jne	620 <_main+0x146f>
100004053: e9 7a 02 00 00              	jmp	634 <_main+0x1482>
100004058: 48 89 c3                    	movq	%rax, %rbx
10000405b: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100004063: 74 1f                       	je	31 <_main+0x1234>
100004065: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
10000406d: e8 2a 2e 00 00              	callq	11818 <dyld_stub_binder+0x100006e9c>
100004072: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
10000407a: 75 16                       	jne	22 <_main+0x1242>
10000407c: e9 df 00 00 00              	jmp	223 <_main+0x1310>
100004081: 48 89 c3                    	movq	%rax, %rbx
100004084: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
10000408c: 0f 84 ce 00 00 00           	je	206 <_main+0x1310>
100004092: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
10000409a: e9 aa 00 00 00              	jmp	170 <_main+0x12f9>
10000409f: 48 89 c3                    	movq	%rax, %rbx
1000040a2: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
1000040aa: 75 23                       	jne	35 <_main+0x127f>
1000040ac: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000040b4: 75 3f                       	jne	63 <_main+0x12a5>
1000040b6: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
1000040be: 75 5b                       	jne	91 <_main+0x12cb>
1000040c0: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
1000040c8: 75 77                       	jne	119 <_main+0x12f1>
1000040ca: e9 91 00 00 00              	jmp	145 <_main+0x1310>
1000040cf: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
1000040d7: e8 c0 2d 00 00              	callq	11712 <dyld_stub_binder+0x100006e9c>
1000040dc: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000040e4: 74 d0                       	je	-48 <_main+0x1266>
1000040e6: eb 0d                       	jmp	13 <_main+0x12a5>
1000040e8: 48 89 c3                    	movq	%rax, %rbx
1000040eb: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000040f3: 74 c1                       	je	-63 <_main+0x1266>
1000040f5: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
1000040fd: e8 9a 2d 00 00              	callq	11674 <dyld_stub_binder+0x100006e9c>
100004102: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
10000410a: 74 b4                       	je	-76 <_main+0x1270>
10000410c: eb 0d                       	jmp	13 <_main+0x12cb>
10000410e: 48 89 c3                    	movq	%rax, %rbx
100004111: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
100004119: 74 a5                       	je	-91 <_main+0x1270>
10000411b: 48 8b bc 24 f0 00 00 00     	movq	240(%rsp), %rdi
100004123: e8 74 2d 00 00              	callq	11636 <dyld_stub_binder+0x100006e9c>
100004128: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100004130: 75 0f                       	jne	15 <_main+0x12f1>
100004132: eb 2c                       	jmp	44 <_main+0x1310>
100004134: 48 89 c3                    	movq	%rax, %rbx
100004137: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
10000413f: 74 1f                       	je	31 <_main+0x1310>
100004141: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
100004149: e8 4e 2d 00 00              	callq	11598 <dyld_stub_binder+0x100006e9c>
10000414e: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
100004156: 48 85 c0                    	testq	%rax, %rax
100004159: 75 12                       	jne	18 <_main+0x131d>
10000415b: eb 20                       	jmp	32 <_main+0x132d>
10000415d: 48 89 c3                    	movq	%rax, %rbx
100004160: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
100004168: 48 85 c0                    	testq	%rax, %rax
10000416b: 74 10                       	je	16 <_main+0x132d>
10000416d: f0                          	lock
10000416e: ff 48 14                    	decl	20(%rax)
100004171: 75 0a                       	jne	10 <_main+0x132d>
100004173: 48 8d 7c 24 78              	leaq	120(%rsp), %rdi
100004178: e8 83 2c 00 00              	callq	11395 <dyld_stub_binder+0x100006e00>
10000417d: 48 c7 84 24 b0 00 00 00 00 00 00 00 	movq	$0, 176(%rsp)
100004189: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
10000418d: 48 8d 44 24 7c              	leaq	124(%rsp), %rax
100004192: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100004197: 83 7c 24 7c 00              	cmpl	$0, 124(%rsp)
10000419c: 7e 1e                       	jle	30 <_main+0x136c>
10000419e: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000041a6: 31 c9                       	xorl	%ecx, %ecx
1000041a8: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000041af: 48 ff c1                    	incq	%rcx
1000041b2: 48 63 54 24 7c              	movslq	124(%rsp), %rdx
1000041b7: 48 39 d1                    	cmpq	%rdx, %rcx
1000041ba: 7c ec                       	jl	-20 <_main+0x1358>
1000041bc: 48 8b bc 24 c0 00 00 00     	movq	192(%rsp), %rdi
1000041c4: 48 8d 84 24 c8 00 00 00     	leaq	200(%rsp), %rax
1000041cc: 48 39 c7                    	cmpq	%rax, %rdi
1000041cf: 74 1f                       	je	31 <_main+0x13a0>
1000041d1: c5 f8 77                    	vzeroupper
1000041d4: e8 5d 2c 00 00              	callq	11357 <dyld_stub_binder+0x100006e36>
1000041d9: eb 15                       	jmp	21 <_main+0x13a0>
1000041db: 48 89 c7                    	movq	%rax, %rdi
1000041de: e8 7d 01 00 00              	callq	381 <_main+0x1510>
1000041e3: eb 08                       	jmp	8 <_main+0x139d>
1000041e5: 48 89 c3                    	movq	%rax, %rbx
1000041e8: e9 8a 00 00 00              	jmp	138 <_main+0x1427>
1000041ed: 48 89 c3                    	movq	%rax, %rbx
1000041f0: 48 8b 84 24 38 01 00 00     	movq	312(%rsp), %rax
1000041f8: 48 85 c0                    	testq	%rax, %rax
1000041fb: 74 16                       	je	22 <_main+0x13c3>
1000041fd: f0                          	lock
1000041fe: ff 48 14                    	decl	20(%rax)
100004201: 75 10                       	jne	16 <_main+0x13c3>
100004203: 48 8d bc 24 00 01 00 00     	leaq	256(%rsp), %rdi
10000420b: c5 f8 77                    	vzeroupper
10000420e: e8 ed 2b 00 00              	callq	11245 <dyld_stub_binder+0x100006e00>
100004213: 48 c7 84 24 38 01 00 00 00 00 00 00 	movq	$0, 312(%rsp)
10000421f: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100004223: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
10000422b: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
10000422f: 83 bc 24 04 01 00 00 00     	cmpl	$0, 260(%rsp)
100004237: 7e 21                       	jle	33 <_main+0x140a>
100004239: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
100004241: 31 c9                       	xorl	%ecx, %ecx
100004243: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
10000424a: 48 ff c1                    	incq	%rcx
10000424d: 48 63 94 24 04 01 00 00     	movslq	260(%rsp), %rdx
100004255: 48 39 d1                    	cmpq	%rdx, %rcx
100004258: 7c e9                       	jl	-23 <_main+0x13f3>
10000425a: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
100004262: 48 8d 84 24 50 01 00 00     	leaq	336(%rsp), %rax
10000426a: 48 39 c7                    	cmpq	%rax, %rdi
10000426d: 74 08                       	je	8 <_main+0x1427>
10000426f: c5 f8 77                    	vzeroupper
100004272: e8 bf 2b 00 00              	callq	11199 <dyld_stub_binder+0x100006e36>
100004277: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
10000427f: c5 f8 77                    	vzeroupper
100004282: e8 e9 00 00 00              	callq	233 <_main+0x1520>
100004287: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000428c: 48 85 c0                    	testq	%rax, %rax
10000428f: 75 2e                       	jne	46 <_main+0x146f>
100004291: eb 3f                       	jmp	63 <_main+0x1482>
100004293: 48 89 c7                    	movq	%rax, %rdi
100004296: e8 c5 00 00 00              	callq	197 <_main+0x1510>
10000429b: 48 89 c3                    	movq	%rax, %rbx
10000429e: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000042a3: 48 85 c0                    	testq	%rax, %rax
1000042a6: 75 17                       	jne	23 <_main+0x146f>
1000042a8: eb 28                       	jmp	40 <_main+0x1482>
1000042aa: 48 89 c7                    	movq	%rax, %rdi
1000042ad: e8 ae 00 00 00              	callq	174 <_main+0x1510>
1000042b2: 48 89 c3                    	movq	%rax, %rbx
1000042b5: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000042ba: 48 85 c0                    	testq	%rax, %rax
1000042bd: 74 13                       	je	19 <_main+0x1482>
1000042bf: f0                          	lock
1000042c0: ff 48 14                    	decl	20(%rax)
1000042c3: 75 0d                       	jne	13 <_main+0x1482>
1000042c5: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
1000042ca: c5 f8 77                    	vzeroupper
1000042cd: e8 2e 2b 00 00              	callq	11054 <dyld_stub_binder+0x100006e00>
1000042d2: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
1000042db: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000042df: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
1000042e4: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
1000042e9: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
1000042ee: 7e 24                       	jle	36 <_main+0x14c4>
1000042f0: 48 8b 44 24 58              	movq	88(%rsp), %rax
1000042f5: 31 c9                       	xorl	%ecx, %ecx
1000042f7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100004300: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004307: 48 ff c1                    	incq	%rcx
10000430a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000430f: 48 39 d1                    	cmpq	%rdx, %rcx
100004312: 7c ec                       	jl	-20 <_main+0x14b0>
100004314: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100004319: 48 8d 44 24 68              	leaq	104(%rsp), %rax
10000431e: 48 39 c7                    	cmpq	%rax, %rdi
100004321: 74 15                       	je	21 <_main+0x14e8>
100004323: c5 f8 77                    	vzeroupper
100004326: e8 0b 2b 00 00              	callq	11019 <dyld_stub_binder+0x100006e36>
10000432b: eb 0b                       	jmp	11 <_main+0x14e8>
10000432d: 48 89 c7                    	movq	%rax, %rdi
100004330: e8 2b 00 00 00              	callq	43 <_main+0x1510>
100004335: 48 89 c3                    	movq	%rax, %rbx
100004338: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100004340: c5 f8 77                    	vzeroupper
100004343: e8 ac 2a 00 00              	callq	10924 <dyld_stub_binder+0x100006df4>
100004348: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100004350: e8 db 05 00 00              	callq	1499 <__ZN14ModelInterfaceD2Ev>
100004355: 48 89 df                    	movq	%rbx, %rdi
100004358: e8 8b 2a 00 00              	callq	10891 <dyld_stub_binder+0x100006de8>
10000435d: 0f 0b                       	ud2
10000435f: 90                          	nop
100004360: 50                          	pushq	%rax
100004361: e8 48 2b 00 00              	callq	11080 <dyld_stub_binder+0x100006eae>
100004366: e8 2b 2b 00 00              	callq	11051 <dyld_stub_binder+0x100006e96>
10000436b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004370: 55                          	pushq	%rbp
100004371: 48 89 e5                    	movq	%rsp, %rbp
100004374: 53                          	pushq	%rbx
100004375: 50                          	pushq	%rax
100004376: 48 89 fb                    	movq	%rdi, %rbx
100004379: 48 8b 87 08 01 00 00        	movq	264(%rdi), %rax
100004380: 48 85 c0                    	testq	%rax, %rax
100004383: 74 12                       	je	18 <_main+0x1547>
100004385: f0                          	lock
100004386: ff 48 14                    	decl	20(%rax)
100004389: 75 0c                       	jne	12 <_main+0x1547>
10000438b: 48 8d bb d0 00 00 00        	leaq	208(%rbx), %rdi
100004392: e8 69 2a 00 00              	callq	10857 <dyld_stub_binder+0x100006e00>
100004397: 48 c7 83 08 01 00 00 00 00 00 00    	movq	$0, 264(%rbx)
1000043a2: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000043a6: c5 fc 11 83 e0 00 00 00     	vmovups	%ymm0, 224(%rbx)
1000043ae: 83 bb d4 00 00 00 00        	cmpl	$0, 212(%rbx)
1000043b5: 7e 1f                       	jle	31 <_main+0x1586>
1000043b7: 48 8b 83 10 01 00 00        	movq	272(%rbx), %rax
1000043be: 31 c9                       	xorl	%ecx, %ecx
1000043c0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000043c7: 48 ff c1                    	incq	%rcx
1000043ca: 48 63 93 d4 00 00 00        	movslq	212(%rbx), %rdx
1000043d1: 48 39 d1                    	cmpq	%rdx, %rcx
1000043d4: 7c ea                       	jl	-22 <_main+0x1570>
1000043d6: 48 8b bb 18 01 00 00        	movq	280(%rbx), %rdi
1000043dd: 48 8d 83 20 01 00 00        	leaq	288(%rbx), %rax
1000043e4: 48 39 c7                    	cmpq	%rax, %rdi
1000043e7: 74 08                       	je	8 <_main+0x15a1>
1000043e9: c5 f8 77                    	vzeroupper
1000043ec: e8 45 2a 00 00              	callq	10821 <dyld_stub_binder+0x100006e36>
1000043f1: 48 8b 83 a8 00 00 00        	movq	168(%rbx), %rax
1000043f8: 48 85 c0                    	testq	%rax, %rax
1000043fb: 74 12                       	je	18 <_main+0x15bf>
1000043fd: f0                          	lock
1000043fe: ff 48 14                    	decl	20(%rax)
100004401: 75 0c                       	jne	12 <_main+0x15bf>
100004403: 48 8d 7b 70                 	leaq	112(%rbx), %rdi
100004407: c5 f8 77                    	vzeroupper
10000440a: e8 f1 29 00 00              	callq	10737 <dyld_stub_binder+0x100006e00>
10000440f: 48 c7 83 a8 00 00 00 00 00 00 00    	movq	$0, 168(%rbx)
10000441a: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000441e: c5 fc 11 83 80 00 00 00     	vmovups	%ymm0, 128(%rbx)
100004426: 83 7b 74 00                 	cmpl	$0, 116(%rbx)
10000442a: 7e 27                       	jle	39 <_main+0x1603>
10000442c: 48 8b 83 b0 00 00 00        	movq	176(%rbx), %rax
100004433: 31 c9                       	xorl	%ecx, %ecx
100004435: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000443f: 90                          	nop
100004440: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004447: 48 ff c1                    	incq	%rcx
10000444a: 48 63 53 74                 	movslq	116(%rbx), %rdx
10000444e: 48 39 d1                    	cmpq	%rdx, %rcx
100004451: 7c ed                       	jl	-19 <_main+0x15f0>
100004453: 48 8b bb b8 00 00 00        	movq	184(%rbx), %rdi
10000445a: 48 8d 83 c0 00 00 00        	leaq	192(%rbx), %rax
100004461: 48 39 c7                    	cmpq	%rax, %rdi
100004464: 74 08                       	je	8 <_main+0x161e>
100004466: c5 f8 77                    	vzeroupper
100004469: e8 c8 29 00 00              	callq	10696 <dyld_stub_binder+0x100006e36>
10000446e: 48 8b 43 48                 	movq	72(%rbx), %rax
100004472: 48 85 c0                    	testq	%rax, %rax
100004475: 74 12                       	je	18 <_main+0x1639>
100004477: f0                          	lock
100004478: ff 48 14                    	decl	20(%rax)
10000447b: 75 0c                       	jne	12 <_main+0x1639>
10000447d: 48 8d 7b 10                 	leaq	16(%rbx), %rdi
100004481: c5 f8 77                    	vzeroupper
100004484: e8 77 29 00 00              	callq	10615 <dyld_stub_binder+0x100006e00>
100004489: 48 c7 43 48 00 00 00 00     	movq	$0, 72(%rbx)
100004491: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004495: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
10000449a: 83 7b 14 00                 	cmpl	$0, 20(%rbx)
10000449e: 7e 23                       	jle	35 <_main+0x1673>
1000044a0: 48 8b 43 50                 	movq	80(%rbx), %rax
1000044a4: 31 c9                       	xorl	%ecx, %ecx
1000044a6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000044b0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000044b7: 48 ff c1                    	incq	%rcx
1000044ba: 48 63 53 14                 	movslq	20(%rbx), %rdx
1000044be: 48 39 d1                    	cmpq	%rdx, %rcx
1000044c1: 7c ed                       	jl	-19 <_main+0x1660>
1000044c3: 48 8b 7b 58                 	movq	88(%rbx), %rdi
1000044c7: 48 83 c3 60                 	addq	$96, %rbx
1000044cb: 48 39 df                    	cmpq	%rbx, %rdi
1000044ce: 74 08                       	je	8 <_main+0x1688>
1000044d0: c5 f8 77                    	vzeroupper
1000044d3: e8 5e 29 00 00              	callq	10590 <dyld_stub_binder+0x100006e36>
1000044d8: 48 83 c4 08                 	addq	$8, %rsp
1000044dc: 5b                          	popq	%rbx
1000044dd: 5d                          	popq	%rbp
1000044de: c5 f8 77                    	vzeroupper
1000044e1: c3                          	retq
1000044e2: 48 89 c7                    	movq	%rax, %rdi
1000044e5: e8 76 fe ff ff              	callq	-394 <_main+0x1510>
1000044ea: 48 89 c7                    	movq	%rax, %rdi
1000044ed: e8 6e fe ff ff              	callq	-402 <_main+0x1510>
1000044f2: 48 89 c7                    	movq	%rax, %rdi
1000044f5: e8 66 fe ff ff              	callq	-410 <_main+0x1510>
1000044fa: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100004500: 55                          	pushq	%rbp
100004501: 48 89 e5                    	movq	%rsp, %rbp
100004504: 41 57                       	pushq	%r15
100004506: 41 56                       	pushq	%r14
100004508: 41 55                       	pushq	%r13
10000450a: 41 54                       	pushq	%r12
10000450c: 53                          	pushq	%rbx
10000450d: 48 83 ec 28                 	subq	$40, %rsp
100004511: 49 89 d6                    	movq	%rdx, %r14
100004514: 49 89 f7                    	movq	%rsi, %r15
100004517: 48 89 fb                    	movq	%rdi, %rbx
10000451a: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
10000451e: 48 89 de                    	movq	%rbx, %rsi
100004521: e8 40 29 00 00              	callq	10560 <dyld_stub_binder+0x100006e66>
100004526: 80 7d b0 00                 	cmpb	$0, -80(%rbp)
10000452a: 0f 84 ae 00 00 00           	je	174 <_main+0x178e>
100004530: 48 8b 03                    	movq	(%rbx), %rax
100004533: 48 8b 40 e8                 	movq	-24(%rax), %rax
100004537: 4c 8d 24 03                 	leaq	(%rbx,%rax), %r12
10000453b: 48 8b 7c 03 28              	movq	40(%rbx,%rax), %rdi
100004540: 44 8b 6c 03 08              	movl	8(%rbx,%rax), %r13d
100004545: 8b 84 03 90 00 00 00        	movl	144(%rbx,%rax), %eax
10000454c: 83 f8 ff                    	cmpl	$-1, %eax
10000454f: 75 4a                       	jne	74 <_main+0x174b>
100004551: 48 89 7d c0                 	movq	%rdi, -64(%rbp)
100004555: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004559: 4c 89 e6                    	movq	%r12, %rsi
10000455c: e8 ed 28 00 00              	callq	10477 <dyld_stub_binder+0x100006e4e>
100004561: 48 8b 35 e8 4a 00 00        	movq	19176(%rip), %rsi
100004568: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
10000456c: e8 d7 28 00 00              	callq	10455 <dyld_stub_binder+0x100006e48>
100004571: 48 8b 08                    	movq	(%rax), %rcx
100004574: 48 89 c7                    	movq	%rax, %rdi
100004577: be 20 00 00 00              	movl	$32, %esi
10000457c: ff 51 38                    	callq	*56(%rcx)
10000457f: 88 45 d7                    	movb	%al, -41(%rbp)
100004582: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004586: e8 ed 28 00 00              	callq	10477 <dyld_stub_binder+0x100006e78>
10000458b: 0f be 45 d7                 	movsbl	-41(%rbp), %eax
10000458f: 41 89 84 24 90 00 00 00     	movl	%eax, 144(%r12)
100004597: 48 8b 7d c0                 	movq	-64(%rbp), %rdi
10000459b: 4d 01 fe                    	addq	%r15, %r14
10000459e: 41 81 e5 b0 00 00 00        	andl	$176, %r13d
1000045a5: 41 83 fd 20                 	cmpl	$32, %r13d
1000045a9: 4c 89 fa                    	movq	%r15, %rdx
1000045ac: 49 0f 44 d6                 	cmoveq	%r14, %rdx
1000045b0: 44 0f be c8                 	movsbl	%al, %r9d
1000045b4: 4c 89 fe                    	movq	%r15, %rsi
1000045b7: 4c 89 f1                    	movq	%r14, %rcx
1000045ba: 4d 89 e0                    	movq	%r12, %r8
1000045bd: e8 9e 00 00 00              	callq	158 <_main+0x1810>
1000045c2: 48 85 c0                    	testq	%rax, %rax
1000045c5: 75 17                       	jne	23 <_main+0x178e>
1000045c7: 48 8b 03                    	movq	(%rbx), %rax
1000045ca: 48 8b 40 e8                 	movq	-24(%rax), %rax
1000045ce: 48 8d 3c 03                 	leaq	(%rbx,%rax), %rdi
1000045d2: 8b 74 03 20                 	movl	32(%rbx,%rax), %esi
1000045d6: 83 ce 05                    	orl	$5, %esi
1000045d9: e8 a6 28 00 00              	callq	10406 <dyld_stub_binder+0x100006e84>
1000045de: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
1000045e2: e8 85 28 00 00              	callq	10373 <dyld_stub_binder+0x100006e6c>
1000045e7: 48 89 d8                    	movq	%rbx, %rax
1000045ea: 48 83 c4 28                 	addq	$40, %rsp
1000045ee: 5b                          	popq	%rbx
1000045ef: 41 5c                       	popq	%r12
1000045f1: 41 5d                       	popq	%r13
1000045f3: 41 5e                       	popq	%r14
1000045f5: 41 5f                       	popq	%r15
1000045f7: 5d                          	popq	%rbp
1000045f8: c3                          	retq
1000045f9: eb 0e                       	jmp	14 <_main+0x17b9>
1000045fb: 49 89 c6                    	movq	%rax, %r14
1000045fe: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004602: e8 71 28 00 00              	callq	10353 <dyld_stub_binder+0x100006e78>
100004607: eb 03                       	jmp	3 <_main+0x17bc>
100004609: 49 89 c6                    	movq	%rax, %r14
10000460c: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100004610: e8 57 28 00 00              	callq	10327 <dyld_stub_binder+0x100006e6c>
100004615: eb 03                       	jmp	3 <_main+0x17ca>
100004617: 49 89 c6                    	movq	%rax, %r14
10000461a: 4c 89 f7                    	movq	%r14, %rdi
10000461d: e8 8c 28 00 00              	callq	10380 <dyld_stub_binder+0x100006eae>
100004622: 48 8b 03                    	movq	(%rbx), %rax
100004625: 48 8b 78 e8                 	movq	-24(%rax), %rdi
100004629: 48 01 df                    	addq	%rbx, %rdi
10000462c: e8 4d 28 00 00              	callq	10317 <dyld_stub_binder+0x100006e7e>
100004631: e8 7e 28 00 00              	callq	10366 <dyld_stub_binder+0x100006eb4>
100004636: eb af                       	jmp	-81 <_main+0x1797>
100004638: 48 89 c3                    	movq	%rax, %rbx
10000463b: e8 74 28 00 00              	callq	10356 <dyld_stub_binder+0x100006eb4>
100004640: 48 89 df                    	movq	%rbx, %rdi
100004643: e8 a0 27 00 00              	callq	10144 <dyld_stub_binder+0x100006de8>
100004648: 0f 0b                       	ud2
10000464a: 48 89 c7                    	movq	%rax, %rdi
10000464d: e8 0e fd ff ff              	callq	-754 <_main+0x1510>
100004652: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000465c: 0f 1f 40 00                 	nopl	(%rax)
100004660: 55                          	pushq	%rbp
100004661: 48 89 e5                    	movq	%rsp, %rbp
100004664: 41 57                       	pushq	%r15
100004666: 41 56                       	pushq	%r14
100004668: 41 55                       	pushq	%r13
10000466a: 41 54                       	pushq	%r12
10000466c: 53                          	pushq	%rbx
10000466d: 48 83 ec 38                 	subq	$56, %rsp
100004671: 48 85 ff                    	testq	%rdi, %rdi
100004674: 0f 84 17 01 00 00           	je	279 <_main+0x1941>
10000467a: 4d 89 c4                    	movq	%r8, %r12
10000467d: 49 89 cf                    	movq	%rcx, %r15
100004680: 49 89 fe                    	movq	%rdi, %r14
100004683: 44 89 4d bc                 	movl	%r9d, -68(%rbp)
100004687: 48 89 c8                    	movq	%rcx, %rax
10000468a: 48 29 f0                    	subq	%rsi, %rax
10000468d: 49 8b 48 18                 	movq	24(%r8), %rcx
100004691: 45 31 ed                    	xorl	%r13d, %r13d
100004694: 48 29 c1                    	subq	%rax, %rcx
100004697: 4c 0f 4f e9                 	cmovgq	%rcx, %r13
10000469b: 48 89 55 a8                 	movq	%rdx, -88(%rbp)
10000469f: 48 89 d3                    	movq	%rdx, %rbx
1000046a2: 48 29 f3                    	subq	%rsi, %rbx
1000046a5: 48 85 db                    	testq	%rbx, %rbx
1000046a8: 7e 15                       	jle	21 <_main+0x186f>
1000046aa: 49 8b 06                    	movq	(%r14), %rax
1000046ad: 4c 89 f7                    	movq	%r14, %rdi
1000046b0: 48 89 da                    	movq	%rbx, %rdx
1000046b3: ff 50 60                    	callq	*96(%rax)
1000046b6: 48 39 d8                    	cmpq	%rbx, %rax
1000046b9: 0f 85 d2 00 00 00           	jne	210 <_main+0x1941>
1000046bf: 4d 85 ed                    	testq	%r13, %r13
1000046c2: 0f 8e a1 00 00 00           	jle	161 <_main+0x1919>
1000046c8: 4c 89 65 b0                 	movq	%r12, -80(%rbp)
1000046cc: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000046d0: c5 f8 29 45 c0              	vmovaps	%xmm0, -64(%rbp)
1000046d5: 48 c7 45 d0 00 00 00 00     	movq	$0, -48(%rbp)
1000046dd: 49 83 fd 17                 	cmpq	$23, %r13
1000046e1: 73 12                       	jae	18 <_main+0x18a5>
1000046e3: 43 8d 44 2d 00              	leal	(%r13,%r13), %eax
1000046e8: 88 45 c0                    	movb	%al, -64(%rbp)
1000046eb: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
1000046ef: 4c 8d 65 c1                 	leaq	-63(%rbp), %r12
1000046f3: eb 27                       	jmp	39 <_main+0x18cc>
1000046f5: 49 8d 5d 10                 	leaq	16(%r13), %rbx
1000046f9: 48 83 e3 f0                 	andq	$-16, %rbx
1000046fd: 48 89 df                    	movq	%rbx, %rdi
100004700: e8 a3 27 00 00              	callq	10147 <dyld_stub_binder+0x100006ea8>
100004705: 49 89 c4                    	movq	%rax, %r12
100004708: 48 89 45 d0                 	movq	%rax, -48(%rbp)
10000470c: 48 83 cb 01                 	orq	$1, %rbx
100004710: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
100004714: 4c 89 6d c8                 	movq	%r13, -56(%rbp)
100004718: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
10000471c: 0f b6 75 bc                 	movzbl	-68(%rbp), %esi
100004720: 4c 89 e7                    	movq	%r12, %rdi
100004723: 4c 89 ea                    	movq	%r13, %rdx
100004726: e8 95 27 00 00              	callq	10133 <dyld_stub_binder+0x100006ec0>
10000472b: 43 c6 04 2c 00              	movb	$0, (%r12,%r13)
100004730: f6 45 c0 01                 	testb	$1, -64(%rbp)
100004734: 74 06                       	je	6 <_main+0x18ec>
100004736: 48 8b 5d d0                 	movq	-48(%rbp), %rbx
10000473a: eb 03                       	jmp	3 <_main+0x18ef>
10000473c: 48 ff c3                    	incq	%rbx
10000473f: 4c 8b 65 b0                 	movq	-80(%rbp), %r12
100004743: 49 8b 06                    	movq	(%r14), %rax
100004746: 4c 89 f7                    	movq	%r14, %rdi
100004749: 48 89 de                    	movq	%rbx, %rsi
10000474c: 4c 89 ea                    	movq	%r13, %rdx
10000474f: ff 50 60                    	callq	*96(%rax)
100004752: 48 89 c3                    	movq	%rax, %rbx
100004755: f6 45 c0 01                 	testb	$1, -64(%rbp)
100004759: 74 09                       	je	9 <_main+0x1914>
10000475b: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
10000475f: e8 38 27 00 00              	callq	10040 <dyld_stub_binder+0x100006e9c>
100004764: 4c 39 eb                    	cmpq	%r13, %rbx
100004767: 75 28                       	jne	40 <_main+0x1941>
100004769: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
10000476d: 49 29 f7                    	subq	%rsi, %r15
100004770: 4d 85 ff                    	testq	%r15, %r15
100004773: 7e 11                       	jle	17 <_main+0x1936>
100004775: 49 8b 06                    	movq	(%r14), %rax
100004778: 4c 89 f7                    	movq	%r14, %rdi
10000477b: 4c 89 fa                    	movq	%r15, %rdx
10000477e: ff 50 60                    	callq	*96(%rax)
100004781: 4c 39 f8                    	cmpq	%r15, %rax
100004784: 75 0b                       	jne	11 <_main+0x1941>
100004786: 49 c7 44 24 18 00 00 00 00  	movq	$0, 24(%r12)
10000478f: eb 03                       	jmp	3 <_main+0x1944>
100004791: 45 31 f6                    	xorl	%r14d, %r14d
100004794: 4c 89 f0                    	movq	%r14, %rax
100004797: 48 83 c4 38                 	addq	$56, %rsp
10000479b: 5b                          	popq	%rbx
10000479c: 41 5c                       	popq	%r12
10000479e: 41 5d                       	popq	%r13
1000047a0: 41 5e                       	popq	%r14
1000047a2: 41 5f                       	popq	%r15
1000047a4: 5d                          	popq	%rbp
1000047a5: c3                          	retq
1000047a6: 48 89 c3                    	movq	%rax, %rbx
1000047a9: f6 45 c0 01                 	testb	$1, -64(%rbp)
1000047ad: 74 09                       	je	9 <_main+0x1968>
1000047af: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
1000047b3: e8 e4 26 00 00              	callq	9956 <dyld_stub_binder+0x100006e9c>
1000047b8: 48 89 df                    	movq	%rbx, %rdi
1000047bb: e8 28 26 00 00              	callq	9768 <dyld_stub_binder+0x100006de8>
1000047c0: 0f 0b                       	ud2
1000047c2: 90                          	nop
1000047c3: 90                          	nop
1000047c4: 90                          	nop
1000047c5: 90                          	nop
1000047c6: 90                          	nop
1000047c7: 90                          	nop
1000047c8: 90                          	nop
1000047c9: 90                          	nop
1000047ca: 90                          	nop
1000047cb: 90                          	nop
1000047cc: 90                          	nop
1000047cd: 90                          	nop
1000047ce: 90                          	nop
1000047cf: 90                          	nop
1000047d0: 55                          	pushq	%rbp
1000047d1: 48 89 e5                    	movq	%rsp, %rbp
1000047d4: 48 8b 05 25 48 00 00        	movq	18469(%rip), %rax
1000047db: 80 38 00                    	cmpb	$0, (%rax)
1000047de: 74 02                       	je	2 <_main+0x1992>
1000047e0: 5d                          	popq	%rbp
1000047e1: c3                          	retq
1000047e2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000047e9: 5d                          	popq	%rbp
1000047ea: c3                          	retq
1000047eb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000047f0: 55                          	pushq	%rbp
1000047f1: 48 89 e5                    	movq	%rsp, %rbp
1000047f4: 48 8b 05 25 48 00 00        	movq	18469(%rip), %rax
1000047fb: 80 38 00                    	cmpb	$0, (%rax)
1000047fe: 74 02                       	je	2 <_main+0x19b2>
100004800: 5d                          	popq	%rbp
100004801: c3                          	retq
100004802: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004809: 5d                          	popq	%rbp
10000480a: c3                          	retq
10000480b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004810: 55                          	pushq	%rbp
100004811: 48 89 e5                    	movq	%rsp, %rbp
100004814: 48 8b 05 1d 48 00 00        	movq	18461(%rip), %rax
10000481b: 80 38 00                    	cmpb	$0, (%rax)
10000481e: 74 02                       	je	2 <_main+0x19d2>
100004820: 5d                          	popq	%rbp
100004821: c3                          	retq
100004822: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004829: 5d                          	popq	%rbp
10000482a: c3                          	retq
10000482b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004830: 55                          	pushq	%rbp
100004831: 48 89 e5                    	movq	%rsp, %rbp
100004834: 48 8b 05 f5 47 00 00        	movq	18421(%rip), %rax
10000483b: 80 38 00                    	cmpb	$0, (%rax)
10000483e: 74 02                       	je	2 <_main+0x19f2>
100004840: 5d                          	popq	%rbp
100004841: c3                          	retq
100004842: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004849: 5d                          	popq	%rbp
10000484a: c3                          	retq
10000484b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004850: 55                          	pushq	%rbp
100004851: 48 89 e5                    	movq	%rsp, %rbp
100004854: 48 8b 05 cd 47 00 00        	movq	18381(%rip), %rax
10000485b: 80 38 00                    	cmpb	$0, (%rax)
10000485e: 74 02                       	je	2 <_main+0x1a12>
100004860: 5d                          	popq	%rbp
100004861: c3                          	retq
100004862: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004869: 5d                          	popq	%rbp
10000486a: c3                          	retq
10000486b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004870: 55                          	pushq	%rbp
100004871: 48 89 e5                    	movq	%rsp, %rbp
100004874: 48 8b 05 8d 47 00 00        	movq	18317(%rip), %rax
10000487b: 80 38 00                    	cmpb	$0, (%rax)
10000487e: 74 02                       	je	2 <_main+0x1a32>
100004880: 5d                          	popq	%rbp
100004881: c3                          	retq
100004882: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004889: 5d                          	popq	%rbp
10000488a: c3                          	retq
10000488b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004890: 55                          	pushq	%rbp
100004891: 48 89 e5                    	movq	%rsp, %rbp
100004894: 48 8b 05 75 47 00 00        	movq	18293(%rip), %rax
10000489b: 80 38 00                    	cmpb	$0, (%rax)
10000489e: 74 02                       	je	2 <_main+0x1a52>
1000048a0: 5d                          	popq	%rbp
1000048a1: c3                          	retq
1000048a2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048a9: 5d                          	popq	%rbp
1000048aa: c3                          	retq
1000048ab: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048b0: 55                          	pushq	%rbp
1000048b1: 48 89 e5                    	movq	%rsp, %rbp
1000048b4: 48 8b 05 85 47 00 00        	movq	18309(%rip), %rax
1000048bb: 80 38 00                    	cmpb	$0, (%rax)
1000048be: 74 02                       	je	2 <_main+0x1a72>
1000048c0: 5d                          	popq	%rbp
1000048c1: c3                          	retq
1000048c2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048c9: 5d                          	popq	%rbp
1000048ca: c3                          	retq
1000048cb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048d0: 55                          	pushq	%rbp
1000048d1: 48 89 e5                    	movq	%rsp, %rbp
1000048d4: 48 8b 05 3d 47 00 00        	movq	18237(%rip), %rax
1000048db: 80 38 00                    	cmpb	$0, (%rax)
1000048de: 74 02                       	je	2 <_main+0x1a92>
1000048e0: 5d                          	popq	%rbp
1000048e1: c3                          	retq
1000048e2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048e9: 5d                          	popq	%rbp
1000048ea: c3                          	retq
1000048eb: 90                          	nop
1000048ec: 90                          	nop
1000048ed: 90                          	nop
1000048ee: 90                          	nop
1000048ef: 90                          	nop

00000001000048f0 __ZN14ModelInterfaceC2Ev:
1000048f0: 55                          	pushq	%rbp
1000048f1: 48 89 e5                    	movq	%rsp, %rbp
1000048f4: 48 8d 05 cd 47 00 00        	leaq	18381(%rip), %rax
1000048fb: 48 89 07                    	movq	%rax, (%rdi)
1000048fe: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004902: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
100004907: 5d                          	popq	%rbp
100004908: c3                          	retq
100004909: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004910 __ZN14ModelInterfaceC1Ev:
100004910: 55                          	pushq	%rbp
100004911: 48 89 e5                    	movq	%rsp, %rbp
100004914: 48 8d 05 ad 47 00 00        	leaq	18349(%rip), %rax
10000491b: 48 89 07                    	movq	%rax, (%rdi)
10000491e: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004922: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
100004927: 5d                          	popq	%rbp
100004928: c3                          	retq
100004929: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004930 __ZN14ModelInterfaceD2Ev:
100004930: 55                          	pushq	%rbp
100004931: 48 89 e5                    	movq	%rsp, %rbp
100004934: 53                          	pushq	%rbx
100004935: 50                          	pushq	%rax
100004936: 48 89 fb                    	movq	%rdi, %rbx
100004939: 48 8d 05 88 47 00 00        	leaq	18312(%rip), %rax
100004940: 48 89 07                    	movq	%rax, (%rdi)
100004943: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100004947: 48 85 ff                    	testq	%rdi, %rdi
10000494a: 74 05                       	je	5 <__ZN14ModelInterfaceD2Ev+0x21>
10000494c: e8 4b 25 00 00              	callq	9547 <dyld_stub_binder+0x100006e9c>
100004951: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100004955: 48 83 c4 08                 	addq	$8, %rsp
100004959: 48 85 ff                    	testq	%rdi, %rdi
10000495c: 74 07                       	je	7 <__ZN14ModelInterfaceD2Ev+0x35>
10000495e: 5b                          	popq	%rbx
10000495f: 5d                          	popq	%rbp
100004960: e9 37 25 00 00              	jmp	9527 <dyld_stub_binder+0x100006e9c>
100004965: 5b                          	popq	%rbx
100004966: 5d                          	popq	%rbp
100004967: c3                          	retq
100004968: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100004970 __ZN14ModelInterfaceD1Ev:
100004970: 55                          	pushq	%rbp
100004971: 48 89 e5                    	movq	%rsp, %rbp
100004974: 53                          	pushq	%rbx
100004975: 50                          	pushq	%rax
100004976: 48 89 fb                    	movq	%rdi, %rbx
100004979: 48 8d 05 48 47 00 00        	leaq	18248(%rip), %rax
100004980: 48 89 07                    	movq	%rax, (%rdi)
100004983: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100004987: 48 85 ff                    	testq	%rdi, %rdi
10000498a: 74 05                       	je	5 <__ZN14ModelInterfaceD1Ev+0x21>
10000498c: e8 0b 25 00 00              	callq	9483 <dyld_stub_binder+0x100006e9c>
100004991: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100004995: 48 83 c4 08                 	addq	$8, %rsp
100004999: 48 85 ff                    	testq	%rdi, %rdi
10000499c: 74 07                       	je	7 <__ZN14ModelInterfaceD1Ev+0x35>
10000499e: 5b                          	popq	%rbx
10000499f: 5d                          	popq	%rbp
1000049a0: e9 f7 24 00 00              	jmp	9463 <dyld_stub_binder+0x100006e9c>
1000049a5: 5b                          	popq	%rbx
1000049a6: 5d                          	popq	%rbp
1000049a7: c3                          	retq
1000049a8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

00000001000049b0 __ZN14ModelInterfaceD0Ev:
1000049b0: 55                          	pushq	%rbp
1000049b1: 48 89 e5                    	movq	%rsp, %rbp
1000049b4: 53                          	pushq	%rbx
1000049b5: 50                          	pushq	%rax
1000049b6: 48 89 fb                    	movq	%rdi, %rbx
1000049b9: 48 8d 05 08 47 00 00        	leaq	18184(%rip), %rax
1000049c0: 48 89 07                    	movq	%rax, (%rdi)
1000049c3: 48 8b 7f 28                 	movq	40(%rdi), %rdi
1000049c7: 48 85 ff                    	testq	%rdi, %rdi
1000049ca: 74 05                       	je	5 <__ZN14ModelInterfaceD0Ev+0x21>
1000049cc: e8 cb 24 00 00              	callq	9419 <dyld_stub_binder+0x100006e9c>
1000049d1: 48 8b 7b 30                 	movq	48(%rbx), %rdi
1000049d5: 48 85 ff                    	testq	%rdi, %rdi
1000049d8: 74 05                       	je	5 <__ZN14ModelInterfaceD0Ev+0x2f>
1000049da: e8 bd 24 00 00              	callq	9405 <dyld_stub_binder+0x100006e9c>
1000049df: 48 89 df                    	movq	%rbx, %rdi
1000049e2: 48 83 c4 08                 	addq	$8, %rsp
1000049e6: 5b                          	popq	%rbx
1000049e7: 5d                          	popq	%rbp
1000049e8: e9 af 24 00 00              	jmp	9391 <dyld_stub_binder+0x100006e9c>
1000049ed: 0f 1f 00                    	nopl	(%rax)

00000001000049f0 __ZN14ModelInterface7forwardEv:
1000049f0: 55                          	pushq	%rbp
1000049f1: 48 89 e5                    	movq	%rsp, %rbp
1000049f4: 5d                          	popq	%rbp
1000049f5: c3                          	retq
1000049f6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

0000000100004a00 __ZN14ModelInterface12input_bufferEv:
100004a00: 55                          	pushq	%rbp
100004a01: 48 89 e5                    	movq	%rsp, %rbp
100004a04: 0f b6 47 24                 	movzbl	36(%rdi), %eax
100004a08: 48 8b 44 c7 28              	movq	40(%rdi,%rax,8), %rax
100004a0d: 5d                          	popq	%rbp
100004a0e: c3                          	retq
100004a0f: 90                          	nop

0000000100004a10 __ZN14ModelInterface13output_bufferEv:
100004a10: 55                          	pushq	%rbp
100004a11: 48 89 e5                    	movq	%rsp, %rbp
100004a14: 31 c0                       	xorl	%eax, %eax
100004a16: 80 7f 24 00                 	cmpb	$0, 36(%rdi)
100004a1a: 0f 94 c0                    	sete	%al
100004a1d: 48 8b 44 c7 28              	movq	40(%rdi,%rax,8), %rax
100004a22: 5d                          	popq	%rbp
100004a23: c3                          	retq
100004a24: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004a2e: 66 90                       	nop

0000000100004a30 __ZN14ModelInterface11init_bufferEj:
100004a30: 55                          	pushq	%rbp
100004a31: 48 89 e5                    	movq	%rsp, %rbp
100004a34: 41 57                       	pushq	%r15
100004a36: 41 56                       	pushq	%r14
100004a38: 41 54                       	pushq	%r12
100004a3a: 53                          	pushq	%rbx
100004a3b: 41 89 f7                    	movl	%esi, %r15d
100004a3e: 48 89 fb                    	movq	%rdi, %rbx
100004a41: c6 47 24 00                 	movb	$0, 36(%rdi)
100004a45: 41 89 f6                    	movl	%esi, %r14d
100004a48: 4c 89 f7                    	movq	%r14, %rdi
100004a4b: e8 52 24 00 00              	callq	9298 <dyld_stub_binder+0x100006ea2>
100004a50: 49 89 c4                    	movq	%rax, %r12
100004a53: 48 89 43 28                 	movq	%rax, 40(%rbx)
100004a57: 4c 89 f7                    	movq	%r14, %rdi
100004a5a: e8 43 24 00 00              	callq	9283 <dyld_stub_binder+0x100006ea2>
100004a5f: 48 89 43 30                 	movq	%rax, 48(%rbx)
100004a63: 45 85 ff                    	testl	%r15d, %r15d
100004a66: 0f 84 44 01 00 00           	je	324 <__ZN14ModelInterface11init_bufferEj+0x180>
100004a6c: 41 c6 04 24 00              	movb	$0, (%r12)
100004a71: 41 83 ff 01                 	cmpl	$1, %r15d
100004a75: 0f 84 95 00 00 00           	je	149 <__ZN14ModelInterface11init_bufferEj+0xe0>
100004a7b: 41 8d 46 ff                 	leal	-1(%r14), %eax
100004a7f: 49 8d 56 fe                 	leaq	-2(%r14), %rdx
100004a83: 83 e0 07                    	andl	$7, %eax
100004a86: b9 01 00 00 00              	movl	$1, %ecx
100004a8b: 48 83 fa 07                 	cmpq	$7, %rdx
100004a8f: 72 63                       	jb	99 <__ZN14ModelInterface11init_bufferEj+0xc4>
100004a91: 48 89 c2                    	movq	%rax, %rdx
100004a94: 48 f7 d2                    	notq	%rdx
100004a97: 4c 01 f2                    	addq	%r14, %rdx
100004a9a: 31 c9                       	xorl	%ecx, %ecx
100004a9c: 0f 1f 40 00                 	nopl	(%rax)
100004aa0: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004aa4: c6 44 0e 01 00              	movb	$0, 1(%rsi,%rcx)
100004aa9: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004aad: c6 44 0e 02 00              	movb	$0, 2(%rsi,%rcx)
100004ab2: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004ab6: c6 44 0e 03 00              	movb	$0, 3(%rsi,%rcx)
100004abb: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004abf: c6 44 0e 04 00              	movb	$0, 4(%rsi,%rcx)
100004ac4: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004ac8: c6 44 0e 05 00              	movb	$0, 5(%rsi,%rcx)
100004acd: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004ad1: c6 44 0e 06 00              	movb	$0, 6(%rsi,%rcx)
100004ad6: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004ada: c6 44 0e 07 00              	movb	$0, 7(%rsi,%rcx)
100004adf: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004ae3: c6 44 0e 08 00              	movb	$0, 8(%rsi,%rcx)
100004ae8: 48 83 c1 08                 	addq	$8, %rcx
100004aec: 48 39 ca                    	cmpq	%rcx, %rdx
100004aef: 75 af                       	jne	-81 <__ZN14ModelInterface11init_bufferEj+0x70>
100004af1: 48 ff c1                    	incq	%rcx
100004af4: 48 85 c0                    	testq	%rax, %rax
100004af7: 74 17                       	je	23 <__ZN14ModelInterface11init_bufferEj+0xe0>
100004af9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004b00: 48 8b 53 28                 	movq	40(%rbx), %rdx
100004b04: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004b08: 48 ff c1                    	incq	%rcx
100004b0b: 48 ff c8                    	decq	%rax
100004b0e: 75 f0                       	jne	-16 <__ZN14ModelInterface11init_bufferEj+0xd0>
100004b10: 45 85 ff                    	testl	%r15d, %r15d
100004b13: 0f 84 97 00 00 00           	je	151 <__ZN14ModelInterface11init_bufferEj+0x180>
100004b19: 49 8d 4e ff                 	leaq	-1(%r14), %rcx
100004b1d: 44 89 f0                    	movl	%r14d, %eax
100004b20: 83 e0 07                    	andl	$7, %eax
100004b23: 48 83 f9 07                 	cmpq	$7, %rcx
100004b27: 73 0c                       	jae	12 <__ZN14ModelInterface11init_bufferEj+0x105>
100004b29: 31 c9                       	xorl	%ecx, %ecx
100004b2b: 48 85 c0                    	testq	%rax, %rax
100004b2e: 75 70                       	jne	112 <__ZN14ModelInterface11init_bufferEj+0x170>
100004b30: e9 7b 00 00 00              	jmp	123 <__ZN14ModelInterface11init_bufferEj+0x180>
100004b35: 49 29 c6                    	subq	%rax, %r14
100004b38: 31 c9                       	xorl	%ecx, %ecx
100004b3a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100004b40: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b44: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004b48: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b4c: c6 44 0a 01 00              	movb	$0, 1(%rdx,%rcx)
100004b51: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b55: c6 44 0a 02 00              	movb	$0, 2(%rdx,%rcx)
100004b5a: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b5e: c6 44 0a 03 00              	movb	$0, 3(%rdx,%rcx)
100004b63: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b67: c6 44 0a 04 00              	movb	$0, 4(%rdx,%rcx)
100004b6c: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b70: c6 44 0a 05 00              	movb	$0, 5(%rdx,%rcx)
100004b75: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b79: c6 44 0a 06 00              	movb	$0, 6(%rdx,%rcx)
100004b7e: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b82: c6 44 0a 07 00              	movb	$0, 7(%rdx,%rcx)
100004b87: 48 83 c1 08                 	addq	$8, %rcx
100004b8b: 49 39 ce                    	cmpq	%rcx, %r14
100004b8e: 75 b0                       	jne	-80 <__ZN14ModelInterface11init_bufferEj+0x110>
100004b90: 48 85 c0                    	testq	%rax, %rax
100004b93: 74 1b                       	je	27 <__ZN14ModelInterface11init_bufferEj+0x180>
100004b95: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004b9f: 90                          	nop
100004ba0: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004ba4: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004ba8: 48 ff c1                    	incq	%rcx
100004bab: 48 ff c8                    	decq	%rax
100004bae: 75 f0                       	jne	-16 <__ZN14ModelInterface11init_bufferEj+0x170>
100004bb0: 5b                          	popq	%rbx
100004bb1: 41 5c                       	popq	%r12
100004bb3: 41 5e                       	popq	%r14
100004bb5: 41 5f                       	popq	%r15
100004bb7: 5d                          	popq	%rbp
100004bb8: c3                          	retq
100004bb9: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004bc0 __ZN14ModelInterface11swap_bufferEv:
100004bc0: 55                          	pushq	%rbp
100004bc1: 48 89 e5                    	movq	%rsp, %rbp
100004bc4: 80 77 24 01                 	xorb	$1, 36(%rdi)
100004bc8: 5d                          	popq	%rbp
100004bc9: c3                          	retq
100004bca: 90                          	nop
100004bcb: 90                          	nop
100004bcc: 90                          	nop
100004bcd: 90                          	nop
100004bce: 90                          	nop
100004bcf: 90                          	nop

0000000100004bd0 __Z4ReLUPaS_j:
100004bd0: 55                          	pushq	%rbp
100004bd1: 48 89 e5                    	movq	%rsp, %rbp
100004bd4: 83 fa 04                    	cmpl	$4, %edx
100004bd7: 0f 82 88 00 00 00           	jb	136 <__Z4ReLUPaS_j+0x95>
100004bdd: 8d 42 fc                    	leal	-4(%rdx), %eax
100004be0: 41 89 c2                    	movl	%eax, %r10d
100004be3: 41 c1 ea 02                 	shrl	$2, %r10d
100004be7: 41 ff c2                    	incl	%r10d
100004bea: 41 83 fa 1f                 	cmpl	$31, %r10d
100004bee: 76 24                       	jbe	36 <__Z4ReLUPaS_j+0x44>
100004bf0: 83 e0 fc                    	andl	$-4, %eax
100004bf3: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100004bf7: 48 83 c1 04                 	addq	$4, %rcx
100004bfb: 48 39 f9                    	cmpq	%rdi, %rcx
100004bfe: 0f 86 78 02 00 00           	jbe	632 <__Z4ReLUPaS_j+0x2ac>
100004c04: 48 01 f8                    	addq	%rdi, %rax
100004c07: 48 83 c0 04                 	addq	$4, %rax
100004c0b: 48 39 f0                    	cmpq	%rsi, %rax
100004c0e: 0f 86 68 02 00 00           	jbe	616 <__Z4ReLUPaS_j+0x2ac>
100004c14: 89 d0                       	movl	%edx, %eax
100004c16: 45 31 c0                    	xorl	%r8d, %r8d
100004c19: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004c20: 0f b6 0e                    	movzbl	(%rsi), %ecx
100004c23: 84 c9                       	testb	%cl, %cl
100004c25: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c29: 88 0f                       	movb	%cl, (%rdi)
100004c2b: 0f b6 4e 01                 	movzbl	1(%rsi), %ecx
100004c2f: 84 c9                       	testb	%cl, %cl
100004c31: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c35: 88 4f 01                    	movb	%cl, 1(%rdi)
100004c38: 0f b6 4e 02                 	movzbl	2(%rsi), %ecx
100004c3c: 84 c9                       	testb	%cl, %cl
100004c3e: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c42: 88 4f 02                    	movb	%cl, 2(%rdi)
100004c45: 0f b6 4e 03                 	movzbl	3(%rsi), %ecx
100004c49: 84 c9                       	testb	%cl, %cl
100004c4b: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c4f: 88 4f 03                    	movb	%cl, 3(%rdi)
100004c52: 48 83 c7 04                 	addq	$4, %rdi
100004c56: 48 83 c6 04                 	addq	$4, %rsi
100004c5a: 83 c0 fc                    	addl	$-4, %eax
100004c5d: 83 f8 03                    	cmpl	$3, %eax
100004c60: 77 be                       	ja	-66 <__Z4ReLUPaS_j+0x50>
100004c62: 83 e2 03                    	andl	$3, %edx
100004c65: 85 d2                       	testl	%edx, %edx
100004c67: 0f 84 0a 02 00 00           	je	522 <__Z4ReLUPaS_j+0x2a7>
100004c6d: 8d 42 ff                    	leal	-1(%rdx), %eax
100004c70: 4c 8d 50 01                 	leaq	1(%rax), %r10
100004c74: 49 83 fa 7f                 	cmpq	$127, %r10
100004c78: 0f 86 2e 01 00 00           	jbe	302 <__Z4ReLUPaS_j+0x1dc>
100004c7e: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100004c82: 48 83 c1 01                 	addq	$1, %rcx
100004c86: 48 39 cf                    	cmpq	%rcx, %rdi
100004c89: 73 10                       	jae	16 <__Z4ReLUPaS_j+0xcb>
100004c8b: 48 01 f8                    	addq	%rdi, %rax
100004c8e: 48 83 c0 01                 	addq	$1, %rax
100004c92: 48 39 c6                    	cmpq	%rax, %rsi
100004c95: 0f 82 11 01 00 00           	jb	273 <__Z4ReLUPaS_j+0x1dc>
100004c9b: 4d 89 d0                    	movq	%r10, %r8
100004c9e: 49 83 e0 80                 	andq	$-128, %r8
100004ca2: 49 8d 40 80                 	leaq	-128(%r8), %rax
100004ca6: 48 89 c1                    	movq	%rax, %rcx
100004ca9: 48 c1 e9 07                 	shrq	$7, %rcx
100004cad: 48 ff c1                    	incq	%rcx
100004cb0: 41 89 c9                    	movl	%ecx, %r9d
100004cb3: 41 83 e1 01                 	andl	$1, %r9d
100004cb7: 48 85 c0                    	testq	%rax, %rax
100004cba: 0f 84 0f 09 00 00           	je	2319 <__Z4ReLUPaS_j+0x9ff>
100004cc0: 4c 89 c8                    	movq	%r9, %rax
100004cc3: 48 29 c8                    	subq	%rcx, %rax
100004cc6: 31 c9                       	xorl	%ecx, %ecx
100004cc8: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100004ccc: 0f 1f 40 00                 	nopl	(%rax)
100004cd0: c4 e2 7d 3c 0c 0e           	vpmaxsb	(%rsi,%rcx), %ymm0, %ymm1
100004cd6: c4 e2 7d 3c 54 0e 20        	vpmaxsb	32(%rsi,%rcx), %ymm0, %ymm2
100004cdd: c4 e2 7d 3c 5c 0e 40        	vpmaxsb	64(%rsi,%rcx), %ymm0, %ymm3
100004ce4: c4 e2 7d 3c 64 0e 60        	vpmaxsb	96(%rsi,%rcx), %ymm0, %ymm4
100004ceb: c5 fe 7f 0c 0f              	vmovdqu	%ymm1, (%rdi,%rcx)
100004cf0: c5 fe 7f 54 0f 20           	vmovdqu	%ymm2, 32(%rdi,%rcx)
100004cf6: c5 fe 7f 5c 0f 40           	vmovdqu	%ymm3, 64(%rdi,%rcx)
100004cfc: c5 fe 7f 64 0f 60           	vmovdqu	%ymm4, 96(%rdi,%rcx)
100004d02: c4 e2 7d 3c 8c 0e 80 00 00 00       	vpmaxsb	128(%rsi,%rcx), %ymm0, %ymm1
100004d0c: c4 e2 7d 3c 94 0e a0 00 00 00       	vpmaxsb	160(%rsi,%rcx), %ymm0, %ymm2
100004d16: c4 e2 7d 3c 9c 0e c0 00 00 00       	vpmaxsb	192(%rsi,%rcx), %ymm0, %ymm3
100004d20: c4 e2 7d 3c a4 0e e0 00 00 00       	vpmaxsb	224(%rsi,%rcx), %ymm0, %ymm4
100004d2a: c5 fe 7f 8c 0f 80 00 00 00  	vmovdqu	%ymm1, 128(%rdi,%rcx)
100004d33: c5 fe 7f 94 0f a0 00 00 00  	vmovdqu	%ymm2, 160(%rdi,%rcx)
100004d3c: c5 fe 7f 9c 0f c0 00 00 00  	vmovdqu	%ymm3, 192(%rdi,%rcx)
100004d45: c5 fe 7f a4 0f e0 00 00 00  	vmovdqu	%ymm4, 224(%rdi,%rcx)
100004d4e: 48 81 c1 00 01 00 00        	addq	$256, %rcx
100004d55: 48 83 c0 02                 	addq	$2, %rax
100004d59: 0f 85 71 ff ff ff           	jne	-143 <__Z4ReLUPaS_j+0x100>
100004d5f: 4d 85 c9                    	testq	%r9, %r9
100004d62: 74 36                       	je	54 <__Z4ReLUPaS_j+0x1ca>
100004d64: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100004d68: c4 e2 7d 3c 0c 0e           	vpmaxsb	(%rsi,%rcx), %ymm0, %ymm1
100004d6e: c4 e2 7d 3c 54 0e 20        	vpmaxsb	32(%rsi,%rcx), %ymm0, %ymm2
100004d75: c4 e2 7d 3c 5c 0e 40        	vpmaxsb	64(%rsi,%rcx), %ymm0, %ymm3
100004d7c: c4 e2 7d 3c 44 0e 60        	vpmaxsb	96(%rsi,%rcx), %ymm0, %ymm0
100004d83: c5 fe 7f 0c 0f              	vmovdqu	%ymm1, (%rdi,%rcx)
100004d88: c5 fe 7f 54 0f 20           	vmovdqu	%ymm2, 32(%rdi,%rcx)
100004d8e: c5 fe 7f 5c 0f 40           	vmovdqu	%ymm3, 64(%rdi,%rcx)
100004d94: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
100004d9a: 4d 39 c2                    	cmpq	%r8, %r10
100004d9d: 0f 84 d4 00 00 00           	je	212 <__Z4ReLUPaS_j+0x2a7>
100004da3: 44 29 c2                    	subl	%r8d, %edx
100004da6: 4c 01 c6                    	addq	%r8, %rsi
100004da9: 4c 01 c7                    	addq	%r8, %rdi
100004dac: 44 8d 42 ff                 	leal	-1(%rdx), %r8d
100004db0: f6 c2 07                    	testb	$7, %dl
100004db3: 74 38                       	je	56 <__Z4ReLUPaS_j+0x21d>
100004db5: 41 89 d2                    	movl	%edx, %r10d
100004db8: 41 83 e2 07                 	andl	$7, %r10d
100004dbc: 45 31 c9                    	xorl	%r9d, %r9d
100004dbf: 31 c9                       	xorl	%ecx, %ecx
100004dc1: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004dcb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004dd0: 0f b6 04 0e                 	movzbl	(%rsi,%rcx), %eax
100004dd4: 84 c0                       	testb	%al, %al
100004dd6: 41 0f 48 c1                 	cmovsl	%r9d, %eax
100004dda: 88 04 0f                    	movb	%al, (%rdi,%rcx)
100004ddd: 48 ff c1                    	incq	%rcx
100004de0: 41 39 ca                    	cmpl	%ecx, %r10d
100004de3: 75 eb                       	jne	-21 <__Z4ReLUPaS_j+0x200>
100004de5: 29 ca                       	subl	%ecx, %edx
100004de7: 48 01 ce                    	addq	%rcx, %rsi
100004dea: 48 01 cf                    	addq	%rcx, %rdi
100004ded: 41 83 f8 07                 	cmpl	$7, %r8d
100004df1: 0f 82 80 00 00 00           	jb	128 <__Z4ReLUPaS_j+0x2a7>
100004df7: 41 89 d0                    	movl	%edx, %r8d
100004dfa: 31 c9                       	xorl	%ecx, %ecx
100004dfc: 31 d2                       	xorl	%edx, %edx
100004dfe: 66 90                       	nop
100004e00: 0f b6 04 16                 	movzbl	(%rsi,%rdx), %eax
100004e04: 84 c0                       	testb	%al, %al
100004e06: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e09: 88 04 17                    	movb	%al, (%rdi,%rdx)
100004e0c: 0f b6 44 16 01              	movzbl	1(%rsi,%rdx), %eax
100004e11: 84 c0                       	testb	%al, %al
100004e13: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e16: 88 44 17 01                 	movb	%al, 1(%rdi,%rdx)
100004e1a: 0f b6 44 16 02              	movzbl	2(%rsi,%rdx), %eax
100004e1f: 84 c0                       	testb	%al, %al
100004e21: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e24: 88 44 17 02                 	movb	%al, 2(%rdi,%rdx)
100004e28: 0f b6 44 16 03              	movzbl	3(%rsi,%rdx), %eax
100004e2d: 84 c0                       	testb	%al, %al
100004e2f: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e32: 88 44 17 03                 	movb	%al, 3(%rdi,%rdx)
100004e36: 0f b6 44 16 04              	movzbl	4(%rsi,%rdx), %eax
100004e3b: 84 c0                       	testb	%al, %al
100004e3d: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e40: 88 44 17 04                 	movb	%al, 4(%rdi,%rdx)
100004e44: 0f b6 44 16 05              	movzbl	5(%rsi,%rdx), %eax
100004e49: 84 c0                       	testb	%al, %al
100004e4b: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e4e: 88 44 17 05                 	movb	%al, 5(%rdi,%rdx)
100004e52: 0f b6 44 16 06              	movzbl	6(%rsi,%rdx), %eax
100004e57: 84 c0                       	testb	%al, %al
100004e59: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e5c: 88 44 17 06                 	movb	%al, 6(%rdi,%rdx)
100004e60: 0f b6 44 16 07              	movzbl	7(%rsi,%rdx), %eax
100004e65: 84 c0                       	testb	%al, %al
100004e67: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e6a: 88 44 17 07                 	movb	%al, 7(%rdi,%rdx)
100004e6e: 48 83 c2 08                 	addq	$8, %rdx
100004e72: 41 39 d0                    	cmpl	%edx, %r8d
100004e75: 75 89                       	jne	-119 <__Z4ReLUPaS_j+0x230>
100004e77: 5d                          	popq	%rbp
100004e78: c5 f8 77                    	vzeroupper
100004e7b: c3                          	retq
100004e7c: 45 89 d0                    	movl	%r10d, %r8d
100004e7f: 41 83 e0 e0                 	andl	$-32, %r8d
100004e83: 49 8d 40 e0                 	leaq	-32(%r8), %rax
100004e87: 48 89 c1                    	movq	%rax, %rcx
100004e8a: 48 c1 e9 05                 	shrq	$5, %rcx
100004e8e: 48 ff c1                    	incq	%rcx
100004e91: 41 89 c9                    	movl	%ecx, %r9d
100004e94: 41 83 e1 01                 	andl	$1, %r9d
100004e98: 48 85 c0                    	testq	%rax, %rax
100004e9b: 0f 84 3e 07 00 00           	je	1854 <__Z4ReLUPaS_j+0xa0f>
100004ea1: 4c 89 c8                    	movq	%r9, %rax
100004ea4: 48 29 c8                    	subq	%rcx, %rax
100004ea7: 31 c9                       	xorl	%ecx, %ecx
100004ea9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004eb0: c5 7a 6f 34 0e              	vmovdqu	(%rsi,%rcx), %xmm14
100004eb5: c5 7a 6f 7c 0e 10           	vmovdqu	16(%rsi,%rcx), %xmm15
100004ebb: c5 fa 6f 54 0e 20           	vmovdqu	32(%rsi,%rcx), %xmm2
100004ec1: c5 fa 6f 5c 0e 30           	vmovdqu	48(%rsi,%rcx), %xmm3
100004ec7: c5 79 6f 1d 51 22 00 00     	vmovdqa	8785(%rip), %xmm11
100004ecf: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100004ed4: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
100004ed9: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100004edd: c5 79 6f 05 4b 22 00 00     	vmovdqa	8779(%rip), %xmm8
100004ee5: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
100004eea: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
100004eef: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004ef3: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
100004ef9: c5 fa 6f 64 0e 70           	vmovdqu	112(%rsi,%rcx), %xmm4
100004eff: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100004f04: c4 e3 fd 00 6c 0e 60 4e     	vpermq	$78, 96(%rsi,%rcx), %ymm5
100004f0c: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004f12: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004f17: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
100004f1b: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100004f21: c5 fa 6f 74 0e 50           	vmovdqu	80(%rsi,%rcx), %xmm6
100004f27: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
100004f2c: c4 63 fd 00 4c 0e 40 4e     	vpermq	$78, 64(%rsi,%rcx), %ymm9
100004f34: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
100004f3a: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
100004f3f: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100004f44: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004f4a: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100004f50: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004f56: c5 79 6f 05 e2 21 00 00     	vmovdqa	8674(%rip), %xmm8
100004f5e: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100004f63: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100004f68: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
100004f6c: c5 79 6f 1d dc 21 00 00     	vmovdqa	8668(%rip), %xmm11
100004f74: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
100004f79: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100004f7e: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
100004f82: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
100004f88: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
100004f8d: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
100004f92: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004f96: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004f9c: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
100004fa1: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
100004fa6: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100004faa: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004fb0: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004fb6: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
100004fbc: c5 79 6f 1d 9c 21 00 00     	vmovdqa	8604(%rip), %xmm11
100004fc4: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100004fc9: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
100004fce: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
100004fd2: c5 f9 6f 0d 96 21 00 00     	vmovdqa	8598(%rip), %xmm1
100004fda: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
100004fde: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100004fe3: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100004fe8: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004fec: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100004ff2: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100004ff7: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004ffc: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005000: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005006: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
10000500b: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100005010: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100005014: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000501a: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005020: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100005026: c5 f9 6f 0d 52 21 00 00     	vmovdqa	8530(%rip), %xmm1
10000502e: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100005033: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100005038: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000503c: c5 f9 6f 15 4c 21 00 00     	vmovdqa	8524(%rip), %xmm2
100005044: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100005048: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
10000504d: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100005052: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005056: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
10000505c: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100005061: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100005066: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000506a: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005070: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
100005075: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
10000507a: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
10000507e: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100005084: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
10000508a: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
100005090: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
100005095: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
10000509a: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
10000509f: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
1000050a4: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000050a9: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
1000050ad: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000050b1: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
1000050b5: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
1000050b9: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000050bd: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000050c1: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000050c5: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000050c9: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
1000050cf: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
1000050d5: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
1000050db: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
1000050e1: c5 fe 7f 4c 0f 40           	vmovdqu	%ymm1, 64(%rdi,%rcx)
1000050e7: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
1000050ed: c5 fe 7f 6c 0f 20           	vmovdqu	%ymm5, 32(%rdi,%rcx)
1000050f3: c5 fe 7f 14 0f              	vmovdqu	%ymm2, (%rdi,%rcx)
1000050f8: c5 7a 6f a4 0e 80 00 00 00  	vmovdqu	128(%rsi,%rcx), %xmm12
100005101: c5 7a 6f ac 0e 90 00 00 00  	vmovdqu	144(%rsi,%rcx), %xmm13
10000510a: c5 7a 6f b4 0e a0 00 00 00  	vmovdqu	160(%rsi,%rcx), %xmm14
100005113: c5 fa 6f 9c 0e b0 00 00 00  	vmovdqu	176(%rsi,%rcx), %xmm3
10000511c: c5 f9 6f 05 fc 1f 00 00     	vmovdqa	8188(%rip), %xmm0
100005124: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
100005129: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
10000512e: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100005132: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100005136: c5 f9 6f 05 f2 1f 00 00     	vmovdqa	8178(%rip), %xmm0
10000513e: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100005143: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
100005148: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
10000514c: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005150: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
100005156: c5 fa 6f a4 0e f0 00 00 00  	vmovdqu	240(%rsi,%rcx), %xmm4
10000515f: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100005164: c4 e3 fd 00 ac 0e e0 00 00 00 4e    	vpermq	$78, 224(%rsi,%rcx), %ymm5
10000516f: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005175: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
10000517a: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000517e: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
100005184: c5 fa 6f b4 0e d0 00 00 00  	vmovdqu	208(%rsi,%rcx), %xmm6
10000518d: c4 e3 fd 00 bc 0e c0 00 00 00 4e    	vpermq	$78, 192(%rsi,%rcx), %ymm7
100005198: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
10000519d: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000051a3: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
1000051a8: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
1000051ac: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000051b2: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
1000051b8: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
1000051be: c5 79 6f 3d 7a 1f 00 00     	vmovdqa	8058(%rip), %xmm15
1000051c6: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
1000051cb: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
1000051d0: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
1000051d4: c5 f9 6f 05 74 1f 00 00     	vmovdqa	8052(%rip), %xmm0
1000051dc: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000051e1: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000051e6: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000051ea: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
1000051f0: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
1000051f5: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
1000051fa: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000051fe: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005204: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005209: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
10000520e: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100005212: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005218: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
10000521e: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005224: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005229: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000522e: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100005232: c5 f9 6f 05 36 1f 00 00     	vmovdqa	7990(%rip), %xmm0
10000523a: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000523f: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005244: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005248: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
10000524e: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005253: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100005258: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000525c: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005261: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005266: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
10000526a: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005270: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005276: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
10000527c: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
100005282: c5 79 6f 3d f6 1e 00 00     	vmovdqa	7926(%rip), %xmm15
10000528a: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
10000528f: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
100005294: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005298: c5 f9 6f 05 f0 1e 00 00     	vmovdqa	7920(%rip), %xmm0
1000052a0: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
1000052a5: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
1000052aa: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000052ae: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
1000052b4: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
1000052b9: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
1000052be: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000052c2: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
1000052c7: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
1000052cc: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000052d0: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000052d6: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
1000052dc: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
1000052e2: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
1000052e8: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
1000052ed: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
1000052f2: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
1000052f7: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000052fc: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100005300: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005304: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005308: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
10000530c: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100005310: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100005314: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005318: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
10000531c: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100005322: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100005328: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
10000532e: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005334: c5 fe 7f 8c 0f c0 00 00 00  	vmovdqu	%ymm1, 192(%rdi,%rcx)
10000533d: c5 fe 7f 84 0f e0 00 00 00  	vmovdqu	%ymm0, 224(%rdi,%rcx)
100005346: c5 fe 7f 9c 0f a0 00 00 00  	vmovdqu	%ymm3, 160(%rdi,%rcx)
10000534f: c5 fe 7f 94 0f 80 00 00 00  	vmovdqu	%ymm2, 128(%rdi,%rcx)
100005358: 48 81 c1 00 01 00 00        	addq	$256, %rcx
10000535f: 48 83 c0 02                 	addq	$2, %rax
100005363: 0f 85 47 fb ff ff           	jne	-1209 <__Z4ReLUPaS_j+0x2e0>
100005369: 4d 85 c9                    	testq	%r9, %r9
10000536c: 0f 84 3e 02 00 00           	je	574 <__Z4ReLUPaS_j+0x9e0>
100005372: c5 7a 6f 14 0e              	vmovdqu	(%rsi,%rcx), %xmm10
100005377: c5 7a 6f 5c 0e 10           	vmovdqu	16(%rsi,%rcx), %xmm11
10000537d: c5 7a 6f 64 0e 20           	vmovdqu	32(%rsi,%rcx), %xmm12
100005383: c5 7a 6f 6c 0e 30           	vmovdqu	48(%rsi,%rcx), %xmm13
100005389: c5 f9 6f 35 8f 1d 00 00     	vmovdqa	7567(%rip), %xmm6
100005391: c4 e2 11 00 e6              	vpshufb	%xmm6, %xmm13, %xmm4
100005396: c4 e2 19 00 ee              	vpshufb	%xmm6, %xmm12, %xmm5
10000539b: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000539f: c5 f9 6f 05 89 1d 00 00     	vmovdqa	7561(%rip), %xmm0
1000053a7: c4 e2 21 00 e8              	vpshufb	%xmm0, %xmm11, %xmm5
1000053ac: c4 e2 29 00 f8              	vpshufb	%xmm0, %xmm10, %xmm7
1000053b1: c5 c1 62 ed                 	vpunpckldq	%xmm5, %xmm7, %xmm5
1000053b5: c4 63 51 02 c4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm8
1000053bb: c5 7a 6f 74 0e 70           	vmovdqu	112(%rsi,%rcx), %xmm14
1000053c1: c4 e2 09 00 fe              	vpshufb	%xmm6, %xmm14, %xmm7
1000053c6: c4 e3 fd 00 6c 0e 60 4e     	vpermq	$78, 96(%rsi,%rcx), %ymm5
1000053ce: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
1000053d4: c4 e2 51 00 f6              	vpshufb	%xmm6, %xmm5, %xmm6
1000053d9: c5 c9 62 f7                 	vpunpckldq	%xmm7, %xmm6, %xmm6
1000053dd: c4 63 7d 38 ce 01           	vinserti128	$1, %xmm6, %ymm0, %ymm9
1000053e3: c5 fa 6f 74 0e 50           	vmovdqu	80(%rsi,%rcx), %xmm6
1000053e9: c4 e2 49 00 c8              	vpshufb	%xmm0, %xmm6, %xmm1
1000053ee: c4 e3 fd 00 7c 0e 40 4e     	vpermq	$78, 64(%rsi,%rcx), %ymm7
1000053f6: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000053fc: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005401: c5 f9 62 c1                 	vpunpckldq	%xmm1, %xmm0, %xmm0
100005405: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000540b: c4 c3 7d 02 c1 c0           	vpblendd	$192, %ymm9, %ymm0, %ymm0
100005411: c4 63 3d 02 c0 f0           	vpblendd	$240, %ymm0, %ymm8, %ymm8
100005417: c5 f9 6f 05 21 1d 00 00     	vmovdqa	7457(%rip), %xmm0
10000541f: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100005424: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005429: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000542d: c5 f9 6f 15 1b 1d 00 00     	vmovdqa	7451(%rip), %xmm2
100005435: c4 e2 21 00 da              	vpshufb	%xmm2, %xmm11, %xmm3
10000543a: c4 e2 29 00 e2              	vpshufb	%xmm2, %xmm10, %xmm4
10000543f: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005443: c4 e3 61 02 c9 0c           	vpblendd	$12, %xmm1, %xmm3, %xmm1
100005449: c4 e2 09 00 d8              	vpshufb	%xmm0, %xmm14, %xmm3
10000544e: c4 e2 51 00 c0              	vpshufb	%xmm0, %xmm5, %xmm0
100005453: c5 f9 62 c3                 	vpunpckldq	%xmm3, %xmm0, %xmm0
100005457: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000545d: c4 e2 49 00 da              	vpshufb	%xmm2, %xmm6, %xmm3
100005462: c4 e2 41 00 d2              	vpshufb	%xmm2, %xmm7, %xmm2
100005467: c5 e9 62 d3                 	vpunpckldq	%xmm3, %xmm2, %xmm2
10000546b: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005471: c4 e3 6d 02 c0 c0           	vpblendd	$192, %ymm0, %ymm2, %ymm0
100005477: c4 63 75 02 c8 f0           	vpblendd	$240, %ymm0, %ymm1, %ymm9
10000547d: c5 f9 6f 05 db 1c 00 00     	vmovdqa	7387(%rip), %xmm0
100005485: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000548a: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
10000548f: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005493: c5 f9 6f 15 d5 1c 00 00     	vmovdqa	7381(%rip), %xmm2
10000549b: c4 e2 21 00 da              	vpshufb	%xmm2, %xmm11, %xmm3
1000054a0: c4 e2 29 00 e2              	vpshufb	%xmm2, %xmm10, %xmm4
1000054a5: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000054a9: c4 e3 61 02 c9 0c           	vpblendd	$12, %xmm1, %xmm3, %xmm1
1000054af: c4 e2 09 00 d8              	vpshufb	%xmm0, %xmm14, %xmm3
1000054b4: c4 e2 51 00 c0              	vpshufb	%xmm0, %xmm5, %xmm0
1000054b9: c5 f9 62 c3                 	vpunpckldq	%xmm3, %xmm0, %xmm0
1000054bd: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000054c3: c4 e2 49 00 da              	vpshufb	%xmm2, %xmm6, %xmm3
1000054c8: c4 e2 41 00 d2              	vpshufb	%xmm2, %xmm7, %xmm2
1000054cd: c5 e9 62 d3                 	vpunpckldq	%xmm3, %xmm2, %xmm2
1000054d1: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000054d7: c4 e3 6d 02 c0 c0           	vpblendd	$192, %ymm0, %ymm2, %ymm0
1000054dd: c4 63 75 02 f8 f0           	vpblendd	$240, %ymm0, %ymm1, %ymm15
1000054e3: c5 f9 6f 0d 95 1c 00 00     	vmovdqa	7317(%rip), %xmm1
1000054eb: c4 e2 11 00 d1              	vpshufb	%xmm1, %xmm13, %xmm2
1000054f0: c4 e2 19 00 d9              	vpshufb	%xmm1, %xmm12, %xmm3
1000054f5: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000054f9: c5 f9 6f 1d 8f 1c 00 00     	vmovdqa	7311(%rip), %xmm3
100005501: c4 e2 21 00 e3              	vpshufb	%xmm3, %xmm11, %xmm4
100005506: c4 e2 29 00 c3              	vpshufb	%xmm3, %xmm10, %xmm0
10000550b: c5 f9 62 c4                 	vpunpckldq	%xmm4, %xmm0, %xmm0
10000550f: c4 e3 79 02 c2 0c           	vpblendd	$12, %xmm2, %xmm0, %xmm0
100005515: c4 e2 09 00 d1              	vpshufb	%xmm1, %xmm14, %xmm2
10000551a: c4 e2 51 00 c9              	vpshufb	%xmm1, %xmm5, %xmm1
10000551f: c5 f1 62 ca                 	vpunpckldq	%xmm2, %xmm1, %xmm1
100005523: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005529: c4 e2 49 00 d3              	vpshufb	%xmm3, %xmm6, %xmm2
10000552e: c4 e2 41 00 db              	vpshufb	%xmm3, %xmm7, %xmm3
100005533: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005537: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
10000553d: c4 e3 6d 02 c9 c0           	vpblendd	$192, %ymm1, %ymm2, %ymm1
100005543: c4 e3 7d 02 c1 f0           	vpblendd	$240, %ymm1, %ymm0, %ymm0
100005549: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
10000554d: c4 e2 3d 3c d1              	vpmaxsb	%ymm1, %ymm8, %ymm2
100005552: c4 e2 35 3c d9              	vpmaxsb	%ymm1, %ymm9, %ymm3
100005557: c4 e2 05 3c e1              	vpmaxsb	%ymm1, %ymm15, %ymm4
10000555c: c4 e2 7d 3c c1              	vpmaxsb	%ymm1, %ymm0, %ymm0
100005561: c5 ed 60 cb                 	vpunpcklbw	%ymm3, %ymm2, %ymm1
100005565: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005569: c5 dd 60 d8                 	vpunpcklbw	%ymm0, %ymm4, %ymm3
10000556d: c5 dd 68 c0                 	vpunpckhbw	%ymm0, %ymm4, %ymm0
100005571: c5 f5 61 e3                 	vpunpcklwd	%ymm3, %ymm1, %ymm4
100005575: c5 f5 69 cb                 	vpunpckhwd	%ymm3, %ymm1, %ymm1
100005579: c5 ed 61 d8                 	vpunpcklwd	%ymm0, %ymm2, %ymm3
10000557d: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100005581: c4 e3 5d 38 d1 01           	vinserti128	$1, %xmm1, %ymm4, %ymm2
100005587: c4 e3 65 38 e8 01           	vinserti128	$1, %xmm0, %ymm3, %ymm5
10000558d: c4 e3 5d 46 c9 31           	vperm2i128	$49, %ymm1, %ymm4, %ymm1
100005593: c4 e3 65 46 c0 31           	vperm2i128	$49, %ymm0, %ymm3, %ymm0
100005599: c5 fe 7f 4c 0f 40           	vmovdqu	%ymm1, 64(%rdi,%rcx)
10000559f: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
1000055a5: c5 fe 7f 6c 0f 20           	vmovdqu	%ymm5, 32(%rdi,%rcx)
1000055ab: c5 fe 7f 14 0f              	vmovdqu	%ymm2, (%rdi,%rcx)
1000055b0: 4a 8d 34 86                 	leaq	(%rsi,%r8,4), %rsi
1000055b4: 4a 8d 3c 87                 	leaq	(%rdi,%r8,4), %rdi
1000055b8: 4d 39 d0                    	cmpq	%r10, %r8
1000055bb: 0f 84 a1 f6 ff ff           	je	-2399 <__Z4ReLUPaS_j+0x92>
1000055c1: 41 c1 e0 02                 	shll	$2, %r8d
1000055c5: 89 d0                       	movl	%edx, %eax
1000055c7: 44 29 c0                    	subl	%r8d, %eax
1000055ca: e9 47 f6 ff ff              	jmp	-2489 <__Z4ReLUPaS_j+0x46>
1000055cf: 31 c9                       	xorl	%ecx, %ecx
1000055d1: 4d 85 c9                    	testq	%r9, %r9
1000055d4: 0f 85 8a f7 ff ff           	jne	-2166 <__Z4ReLUPaS_j+0x194>
1000055da: e9 bb f7 ff ff              	jmp	-2117 <__Z4ReLUPaS_j+0x1ca>
1000055df: 31 c9                       	xorl	%ecx, %ecx
1000055e1: 4d 85 c9                    	testq	%r9, %r9
1000055e4: 0f 85 88 fd ff ff           	jne	-632 <__Z4ReLUPaS_j+0x7a2>
1000055ea: eb c4                       	jmp	-60 <__Z4ReLUPaS_j+0x9e0>
1000055ec: 90                          	nop
1000055ed: 90                          	nop
1000055ee: 90                          	nop
1000055ef: 90                          	nop

00000001000055f0 __ZN11LineNetworkC2Ev:
1000055f0: 55                          	pushq	%rbp
1000055f1: 48 89 e5                    	movq	%rsp, %rbp
1000055f4: 41 56                       	pushq	%r14
1000055f6: 53                          	pushq	%rbx
1000055f7: 48 89 fb                    	movq	%rdi, %rbx
1000055fa: e8 f1 f2 ff ff              	callq	-3343 <__ZN14ModelInterfaceC2Ev>
1000055ff: 48 8d 05 fa 3a 00 00        	leaq	15098(%rip), %rax
100005606: 48 89 03                    	movq	%rax, (%rbx)
100005609: 48 89 df                    	movq	%rbx, %rdi
10000560c: be 00 00 08 00              	movl	$524288, %esi
100005611: e8 1a f4 ff ff              	callq	-3046 <__ZN14ModelInterface11init_bufferEj>
100005616: c7 43 20 00 08 16 03        	movl	$51775488, 32(%rbx)
10000561d: c5 f8 28 05 7b 1b 00 00     	vmovaps	7035(%rip), %xmm0
100005625: c5 f8 11 43 08              	vmovups	%xmm0, 8(%rbx)
10000562a: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
100005634: 48 89 43 18                 	movq	%rax, 24(%rbx)
100005638: 5b                          	popq	%rbx
100005639: 41 5e                       	popq	%r14
10000563b: 5d                          	popq	%rbp
10000563c: c3                          	retq
10000563d: 49 89 c6                    	movq	%rax, %r14
100005640: 48 89 df                    	movq	%rbx, %rdi
100005643: e8 e8 f2 ff ff              	callq	-3352 <__ZN14ModelInterfaceD2Ev>
100005648: 4c 89 f7                    	movq	%r14, %rdi
10000564b: e8 98 17 00 00              	callq	6040 <dyld_stub_binder+0x100006de8>
100005650: 0f 0b                       	ud2
100005652: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000565c: 0f 1f 40 00                 	nopl	(%rax)

0000000100005660 __ZN11LineNetworkC1Ev:
100005660: 55                          	pushq	%rbp
100005661: 48 89 e5                    	movq	%rsp, %rbp
100005664: 41 56                       	pushq	%r14
100005666: 53                          	pushq	%rbx
100005667: 48 89 fb                    	movq	%rdi, %rbx
10000566a: e8 81 f2 ff ff              	callq	-3455 <__ZN14ModelInterfaceC2Ev>
10000566f: 48 8d 05 8a 3a 00 00        	leaq	14986(%rip), %rax
100005676: 48 89 03                    	movq	%rax, (%rbx)
100005679: 48 89 df                    	movq	%rbx, %rdi
10000567c: be 00 00 08 00              	movl	$524288, %esi
100005681: e8 aa f3 ff ff              	callq	-3158 <__ZN14ModelInterface11init_bufferEj>
100005686: c7 43 20 00 08 16 03        	movl	$51775488, 32(%rbx)
10000568d: c5 f8 28 05 0b 1b 00 00     	vmovaps	6923(%rip), %xmm0
100005695: c5 f8 11 43 08              	vmovups	%xmm0, 8(%rbx)
10000569a: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
1000056a4: 48 89 43 18                 	movq	%rax, 24(%rbx)
1000056a8: 5b                          	popq	%rbx
1000056a9: 41 5e                       	popq	%r14
1000056ab: 5d                          	popq	%rbp
1000056ac: c3                          	retq
1000056ad: 49 89 c6                    	movq	%rax, %r14
1000056b0: 48 89 df                    	movq	%rbx, %rdi
1000056b3: e8 78 f2 ff ff              	callq	-3464 <__ZN14ModelInterfaceD2Ev>
1000056b8: 4c 89 f7                    	movq	%r14, %rdi
1000056bb: e8 28 17 00 00              	callq	5928 <dyld_stub_binder+0x100006de8>
1000056c0: 0f 0b                       	ud2
1000056c2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000056cc: 0f 1f 40 00                 	nopl	(%rax)

00000001000056d0 __ZN11LineNetwork7forwardEv:
1000056d0: 55                          	pushq	%rbp
1000056d1: 48 89 e5                    	movq	%rsp, %rbp
1000056d4: 41 57                       	pushq	%r15
1000056d6: 41 56                       	pushq	%r14
1000056d8: 41 55                       	pushq	%r13
1000056da: 41 54                       	pushq	%r12
1000056dc: 53                          	pushq	%rbx
1000056dd: 48 83 ec 58                 	subq	$88, %rsp
1000056e1: 49 89 ff                    	movq	%rdi, %r15
1000056e4: e8 27 f3 ff ff              	callq	-3289 <__ZN14ModelInterface13output_bufferEv>
1000056e9: 49 89 c6                    	movq	%rax, %r14
1000056ec: 4c 89 ff                    	movq	%r15, %rdi
1000056ef: e8 0c f3 ff ff              	callq	-3316 <__ZN14ModelInterface12input_bufferEv>
1000056f4: 48 8d 15 65 1c 00 00        	leaq	7269(%rip), %rdx
1000056fb: 48 8d 0d a6 1c 00 00        	leaq	7334(%rip), %rcx
100005702: 4c 89 f7                    	movq	%r14, %rdi
100005705: 48 89 c6                    	movq	%rax, %rsi
100005708: 41 b8 37 00 00 00           	movl	$55, %r8d
10000570e: e8 6d 05 00 00              	callq	1389 <__ZN11LineNetwork7forwardEv+0x5b0>
100005713: 4c 89 ff                    	movq	%r15, %rdi
100005716: e8 a5 f4 ff ff              	callq	-2907 <__ZN14ModelInterface11swap_bufferEv>
10000571b: 4c 89 ff                    	movq	%r15, %rdi
10000571e: e8 ed f2 ff ff              	callq	-3347 <__ZN14ModelInterface13output_bufferEv>
100005723: 49 89 c6                    	movq	%rax, %r14
100005726: 4c 89 ff                    	movq	%r15, %rdi
100005729: e8 d2 f2 ff ff              	callq	-3374 <__ZN14ModelInterface12input_bufferEv>
10000572e: 4c 89 f7                    	movq	%r14, %rdi
100005731: 48 89 c6                    	movq	%rax, %rsi
100005734: ba 00 00 08 00              	movl	$524288, %edx
100005739: e8 92 f4 ff ff              	callq	-2926 <__Z4ReLUPaS_j>
10000573e: 4c 89 ff                    	movq	%r15, %rdi
100005741: e8 7a f4 ff ff              	callq	-2950 <__ZN14ModelInterface11swap_bufferEv>
100005746: 4c 89 ff                    	movq	%r15, %rdi
100005749: e8 c2 f2 ff ff              	callq	-3390 <__ZN14ModelInterface13output_bufferEv>
10000574e: 49 89 c5                    	movq	%rax, %r13
100005751: 4c 89 7d 88                 	movq	%r15, -120(%rbp)
100005755: 4c 89 ff                    	movq	%r15, %rdi
100005758: e8 a3 f2 ff ff              	callq	-3421 <__ZN14ModelInterface12input_bufferEv>
10000575d: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100005761: 31 c0                       	xorl	%eax, %eax
100005763: eb 23                       	jmp	35 <__ZN11LineNetwork7forwardEv+0xb8>
100005765: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000576f: 90                          	nop
100005770: 48 8b 45 c0                 	movq	-64(%rbp), %rax
100005774: 48 ff c0                    	incq	%rax
100005777: 4c 8b 6d b8                 	movq	-72(%rbp), %r13
10000577b: 49 ff c5                    	incq	%r13
10000577e: 48 83 f8 08                 	cmpq	$8, %rax
100005782: 0f 84 02 01 00 00           	je	258 <__ZN11LineNetwork7forwardEv+0x1ba>
100005788: 48 89 45 c0                 	movq	%rax, -64(%rbp)
10000578c: 48 8d 04 c5 00 00 00 00     	leaq	(,%rax,8), %rax
100005794: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005798: 48 8d 05 11 1c 00 00        	leaq	7185(%rip), %rax
10000579f: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
1000057a3: 48 89 55 90                 	movq	%rdx, -112(%rbp)
1000057a7: 48 8d 54 08 18              	leaq	24(%rax,%rcx), %rdx
1000057ac: 48 89 55 98                 	movq	%rdx, -104(%rbp)
1000057b0: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
1000057b4: 48 8d 44 08 30              	leaq	48(%rax,%rcx), %rax
1000057b9: 48 89 45 a0                 	movq	%rax, -96(%rbp)
1000057bd: 4c 89 6d b8                 	movq	%r13, -72(%rbp)
1000057c1: 4c 8b 7d c8                 	movq	-56(%rbp), %r15
1000057c5: 31 c0                       	xorl	%eax, %eax
1000057c7: eb 25                       	jmp	37 <__ZN11LineNetwork7forwardEv+0x11e>
1000057c9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
1000057d0: 4c 8b 7d a8                 	movq	-88(%rbp), %r15
1000057d4: 49 81 c7 00 10 00 00        	addq	$4096, %r15
1000057db: 49 81 c5 00 04 00 00        	addq	$1024, %r13
1000057e2: 48 8b 45 b0                 	movq	-80(%rbp), %rax
1000057e6: 48 3d fd 00 00 00           	cmpq	$253, %rax
1000057ec: 73 82                       	jae	-126 <__ZN11LineNetwork7forwardEv+0xa0>
1000057ee: 48 83 c0 02                 	addq	$2, %rax
1000057f2: 48 89 45 b0                 	movq	%rax, -80(%rbp)
1000057f6: 4c 89 7d a8                 	movq	%r15, -88(%rbp)
1000057fa: 31 db                       	xorl	%ebx, %ebx
1000057fc: eb 18                       	jmp	24 <__ZN11LineNetwork7forwardEv+0x146>
1000057fe: 66 90                       	nop
100005800: 41 88 44 9d 00              	movb	%al, (%r13,%rbx,4)
100005805: 48 83 c3 02                 	addq	$2, %rbx
100005809: 49 83 c7 10                 	addq	$16, %r15
10000580d: 48 81 fb fd 00 00 00        	cmpq	$253, %rbx
100005814: 73 ba                       	jae	-70 <__ZN11LineNetwork7forwardEv+0x100>
100005816: 4c 89 ff                    	movq	%r15, %rdi
100005819: 48 8b 75 90                 	movq	-112(%rbp), %rsi
10000581d: e8 1e 12 00 00              	callq	4638 <__ZN11LineNetwork7forwardEv+0x1370>
100005822: 41 89 c6                    	movl	%eax, %r14d
100005825: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
10000582c: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100005830: e8 0b 12 00 00              	callq	4619 <__ZN11LineNetwork7forwardEv+0x1370>
100005835: 41 89 c4                    	movl	%eax, %r12d
100005838: 45 01 f4                    	addl	%r14d, %r12d
10000583b: 49 8d bf 00 10 00 00        	leaq	4096(%r15), %rdi
100005842: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005846: e8 f5 11 00 00              	callq	4597 <__ZN11LineNetwork7forwardEv+0x1370>
10000584b: 44 01 e0                    	addl	%r12d, %eax
10000584e: 48 8d 0d 9b 1d 00 00        	leaq	7579(%rip), %rcx
100005855: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
100005859: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
10000585d: 01 c1                       	addl	%eax, %ecx
10000585f: 6b c9 37                    	imull	$55, %ecx, %ecx
100005862: 89 c8                       	movl	%ecx, %eax
100005864: c1 f8 1f                    	sarl	$31, %eax
100005867: c1 e8 12                    	shrl	$18, %eax
10000586a: 01 c8                       	addl	%ecx, %eax
10000586c: c1 f8 0e                    	sarl	$14, %eax
10000586f: 3d 80 00 00 00              	cmpl	$128, %eax
100005874: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x1ab>
100005876: b8 7f 00 00 00              	movl	$127, %eax
10000587b: 83 f8 81                    	cmpl	$-127, %eax
10000587e: 7f 80                       	jg	-128 <__ZN11LineNetwork7forwardEv+0x130>
100005880: b8 81 00 00 00              	movl	$129, %eax
100005885: e9 76 ff ff ff              	jmp	-138 <__ZN11LineNetwork7forwardEv+0x130>
10000588a: 4c 8b 7d 88                 	movq	-120(%rbp), %r15
10000588e: 4c 89 ff                    	movq	%r15, %rdi
100005891: e8 2a f3 ff ff              	callq	-3286 <__ZN14ModelInterface11swap_bufferEv>
100005896: 4c 89 ff                    	movq	%r15, %rdi
100005899: e8 72 f1 ff ff              	callq	-3726 <__ZN14ModelInterface13output_bufferEv>
10000589e: 49 89 c6                    	movq	%rax, %r14
1000058a1: 4c 89 ff                    	movq	%r15, %rdi
1000058a4: e8 57 f1 ff ff              	callq	-3753 <__ZN14ModelInterface12input_bufferEv>
1000058a9: 4c 89 f7                    	movq	%r14, %rdi
1000058ac: 48 89 c6                    	movq	%rax, %rsi
1000058af: ba 00 00 02 00              	movl	$131072, %edx
1000058b4: e8 17 f3 ff ff              	callq	-3305 <__Z4ReLUPaS_j>
1000058b9: 4c 89 ff                    	movq	%r15, %rdi
1000058bc: e8 ff f2 ff ff              	callq	-3329 <__ZN14ModelInterface11swap_bufferEv>
1000058c1: 4c 89 ff                    	movq	%r15, %rdi
1000058c4: e8 47 f1 ff ff              	callq	-3769 <__ZN14ModelInterface13output_bufferEv>
1000058c9: 49 89 c5                    	movq	%rax, %r13
1000058cc: 4c 89 ff                    	movq	%r15, %rdi
1000058cf: e8 2c f1 ff ff              	callq	-3796 <__ZN14ModelInterface12input_bufferEv>
1000058d4: 48 89 45 c8                 	movq	%rax, -56(%rbp)
1000058d8: 31 c0                       	xorl	%eax, %eax
1000058da: eb 1c                       	jmp	28 <__ZN11LineNetwork7forwardEv+0x228>
1000058dc: 0f 1f 40 00                 	nopl	(%rax)
1000058e0: 48 8b 45 c0                 	movq	-64(%rbp), %rax
1000058e4: 48 ff c0                    	incq	%rax
1000058e7: 4c 8b 6d b8                 	movq	-72(%rbp), %r13
1000058eb: 49 ff c5                    	incq	%r13
1000058ee: 48 83 f8 10                 	cmpq	$16, %rax
1000058f2: 0f 84 ff 00 00 00           	je	255 <__ZN11LineNetwork7forwardEv+0x327>
1000058f8: 48 89 45 c0                 	movq	%rax, -64(%rbp)
1000058fc: 48 8d 04 c5 00 00 00 00     	leaq	(,%rax,8), %rax
100005904: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005908: 48 8d 05 f1 1c 00 00        	leaq	7409(%rip), %rax
10000590f: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005913: 48 89 55 90                 	movq	%rdx, -112(%rbp)
100005917: 48 8d 54 08 18              	leaq	24(%rax,%rcx), %rdx
10000591c: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005920: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
100005924: 48 8d 44 08 30              	leaq	48(%rax,%rcx), %rax
100005929: 48 89 45 a0                 	movq	%rax, -96(%rbp)
10000592d: 4c 89 6d b8                 	movq	%r13, -72(%rbp)
100005931: 4c 8b 7d c8                 	movq	-56(%rbp), %r15
100005935: 31 c0                       	xorl	%eax, %eax
100005937: eb 23                       	jmp	35 <__ZN11LineNetwork7forwardEv+0x28c>
100005939: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005940: 4c 8b 7d a8                 	movq	-88(%rbp), %r15
100005944: 49 81 c7 00 08 00 00        	addq	$2048, %r15
10000594b: 49 81 c5 00 04 00 00        	addq	$1024, %r13
100005952: 48 8b 45 b0                 	movq	-80(%rbp), %rax
100005956: 48 83 f8 7d                 	cmpq	$125, %rax
10000595a: 73 84                       	jae	-124 <__ZN11LineNetwork7forwardEv+0x210>
10000595c: 48 83 c0 02                 	addq	$2, %rax
100005960: 48 89 45 b0                 	movq	%rax, -80(%rbp)
100005964: 4c 89 7d a8                 	movq	%r15, -88(%rbp)
100005968: 31 db                       	xorl	%ebx, %ebx
10000596a: eb 17                       	jmp	23 <__ZN11LineNetwork7forwardEv+0x2b3>
10000596c: 0f 1f 40 00                 	nopl	(%rax)
100005970: 41 88 44 dd 00              	movb	%al, (%r13,%rbx,8)
100005975: 48 83 c3 02                 	addq	$2, %rbx
100005979: 49 83 c7 10                 	addq	$16, %r15
10000597d: 48 83 fb 7d                 	cmpq	$125, %rbx
100005981: 73 bd                       	jae	-67 <__ZN11LineNetwork7forwardEv+0x270>
100005983: 4c 89 ff                    	movq	%r15, %rdi
100005986: 48 8b 75 90                 	movq	-112(%rbp), %rsi
10000598a: e8 b1 10 00 00              	callq	4273 <__ZN11LineNetwork7forwardEv+0x1370>
10000598f: 41 89 c6                    	movl	%eax, %r14d
100005992: 49 8d bf 00 04 00 00        	leaq	1024(%r15), %rdi
100005999: 48 8b 75 98                 	movq	-104(%rbp), %rsi
10000599d: e8 9e 10 00 00              	callq	4254 <__ZN11LineNetwork7forwardEv+0x1370>
1000059a2: 41 89 c4                    	movl	%eax, %r12d
1000059a5: 45 01 f4                    	addl	%r14d, %r12d
1000059a8: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
1000059af: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
1000059b3: e8 88 10 00 00              	callq	4232 <__ZN11LineNetwork7forwardEv+0x1370>
1000059b8: 44 01 e0                    	addl	%r12d, %eax
1000059bb: 48 8d 0d be 20 00 00        	leaq	8382(%rip), %rcx
1000059c2: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
1000059c6: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
1000059ca: 01 c1                       	addl	%eax, %ecx
1000059cc: 6b c9 39                    	imull	$57, %ecx, %ecx
1000059cf: 89 c8                       	movl	%ecx, %eax
1000059d1: c1 f8 1f                    	sarl	$31, %eax
1000059d4: c1 e8 12                    	shrl	$18, %eax
1000059d7: 01 c8                       	addl	%ecx, %eax
1000059d9: c1 f8 0e                    	sarl	$14, %eax
1000059dc: 3d 80 00 00 00              	cmpl	$128, %eax
1000059e1: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x318>
1000059e3: b8 7f 00 00 00              	movl	$127, %eax
1000059e8: 83 f8 81                    	cmpl	$-127, %eax
1000059eb: 7f 83                       	jg	-125 <__ZN11LineNetwork7forwardEv+0x2a0>
1000059ed: b8 81 00 00 00              	movl	$129, %eax
1000059f2: e9 79 ff ff ff              	jmp	-135 <__ZN11LineNetwork7forwardEv+0x2a0>
1000059f7: 4c 8b 7d 88                 	movq	-120(%rbp), %r15
1000059fb: 4c 89 ff                    	movq	%r15, %rdi
1000059fe: e8 bd f1 ff ff              	callq	-3651 <__ZN14ModelInterface11swap_bufferEv>
100005a03: 4c 89 ff                    	movq	%r15, %rdi
100005a06: e8 05 f0 ff ff              	callq	-4091 <__ZN14ModelInterface13output_bufferEv>
100005a0b: 49 89 c6                    	movq	%rax, %r14
100005a0e: 4c 89 ff                    	movq	%r15, %rdi
100005a11: e8 ea ef ff ff              	callq	-4118 <__ZN14ModelInterface12input_bufferEv>
100005a16: 4c 89 f7                    	movq	%r14, %rdi
100005a19: 48 89 c6                    	movq	%rax, %rsi
100005a1c: ba 00 00 01 00              	movl	$65536, %edx
100005a21: e8 aa f1 ff ff              	callq	-3670 <__Z4ReLUPaS_j>
100005a26: 4c 89 ff                    	movq	%r15, %rdi
100005a29: e8 92 f1 ff ff              	callq	-3694 <__ZN14ModelInterface11swap_bufferEv>
100005a2e: 4c 89 ff                    	movq	%r15, %rdi
100005a31: e8 da ef ff ff              	callq	-4134 <__ZN14ModelInterface13output_bufferEv>
100005a36: 48 89 c3                    	movq	%rax, %rbx
100005a39: 4c 89 ff                    	movq	%r15, %rdi
100005a3c: e8 bf ef ff ff              	callq	-4161 <__ZN14ModelInterface12input_bufferEv>
100005a41: 48 89 45 80                 	movq	%rax, -128(%rbp)
100005a45: 31 c0                       	xorl	%eax, %eax
100005a47: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x398>
100005a49: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005a50: 48 8b 45 c8                 	movq	-56(%rbp), %rax
100005a54: 48 ff c0                    	incq	%rax
100005a57: 48 8b 5d c0                 	movq	-64(%rbp), %rbx
100005a5b: 48 ff c3                    	incq	%rbx
100005a5e: 48 83 f8 20                 	cmpq	$32, %rax
100005a62: 0f 84 17 01 00 00           	je	279 <__ZN11LineNetwork7forwardEv+0x4af>
100005a68: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100005a6c: 48 c1 e0 04                 	shlq	$4, %rax
100005a70: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005a74: 48 8d 05 15 20 00 00        	leaq	8213(%rip), %rax
100005a7b: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005a7f: 48 89 55 90                 	movq	%rdx, -112(%rbp)
100005a83: 48 8d 54 08 30              	leaq	48(%rax,%rcx), %rdx
100005a88: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005a8c: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
100005a90: 48 8d 44 08 60              	leaq	96(%rax,%rcx), %rax
100005a95: 48 89 45 a0                 	movq	%rax, -96(%rbp)
100005a99: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
100005a9d: 4c 8b 7d 80                 	movq	-128(%rbp), %r15
100005aa1: 31 c0                       	xorl	%eax, %eax
100005aa3: eb 2b                       	jmp	43 <__ZN11LineNetwork7forwardEv+0x400>
100005aa5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005aaf: 90                          	nop
100005ab0: 4c 8b 7d b0                 	movq	-80(%rbp), %r15
100005ab4: 49 81 c7 00 08 00 00        	addq	$2048, %r15
100005abb: 48 8b 5d a8                 	movq	-88(%rbp), %rbx
100005abf: 48 81 c3 00 04 00 00        	addq	$1024, %rbx
100005ac6: 48 8b 45 b8                 	movq	-72(%rbp), %rax
100005aca: 48 83 f8 3d                 	cmpq	$61, %rax
100005ace: 73 80                       	jae	-128 <__ZN11LineNetwork7forwardEv+0x380>
100005ad0: 48 83 c0 02                 	addq	$2, %rax
100005ad4: 48 89 45 b8                 	movq	%rax, -72(%rbp)
100005ad8: 48 89 5d a8                 	movq	%rbx, -88(%rbp)
100005adc: 4c 89 7d b0                 	movq	%r15, -80(%rbp)
100005ae0: 45 31 f6                    	xorl	%r14d, %r14d
100005ae3: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x434>
100005ae5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005aef: 90                          	nop
100005af0: 88 03                       	movb	%al, (%rbx)
100005af2: 49 83 c6 02                 	addq	$2, %r14
100005af6: 49 83 c7 20                 	addq	$32, %r15
100005afa: 48 83 c3 20                 	addq	$32, %rbx
100005afe: 49 83 fe 3d                 	cmpq	$61, %r14
100005b02: 73 ac                       	jae	-84 <__ZN11LineNetwork7forwardEv+0x3e0>
100005b04: 4c 89 ff                    	movq	%r15, %rdi
100005b07: 48 8b 75 90                 	movq	-112(%rbp), %rsi
100005b0b: e8 b0 10 00 00              	callq	4272 <__ZN11LineNetwork7forwardEv+0x14f0>
100005b10: 41 89 c4                    	movl	%eax, %r12d
100005b13: 49 8d bf 00 04 00 00        	leaq	1024(%r15), %rdi
100005b1a: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100005b1e: e8 9d 10 00 00              	callq	4253 <__ZN11LineNetwork7forwardEv+0x14f0>
100005b23: 41 89 c5                    	movl	%eax, %r13d
100005b26: 45 01 e5                    	addl	%r12d, %r13d
100005b29: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
100005b30: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005b34: e8 87 10 00 00              	callq	4231 <__ZN11LineNetwork7forwardEv+0x14f0>
100005b39: 44 01 e8                    	addl	%r13d, %eax
100005b3c: 48 8d 0d 4d 31 00 00        	leaq	12621(%rip), %rcx
100005b43: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
100005b47: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
100005b4b: 01 c1                       	addl	%eax, %ecx
100005b4d: c1 e1 04                    	shll	$4, %ecx
100005b50: 8d 0c 49                    	leal	(%rcx,%rcx,2), %ecx
100005b53: 89 c8                       	movl	%ecx, %eax
100005b55: c1 f8 1f                    	sarl	$31, %eax
100005b58: c1 e8 12                    	shrl	$18, %eax
100005b5b: 01 c8                       	addl	%ecx, %eax
100005b5d: c1 f8 0e                    	sarl	$14, %eax
100005b60: 3d 80 00 00 00              	cmpl	$128, %eax
100005b65: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x49c>
100005b67: b8 7f 00 00 00              	movl	$127, %eax
100005b6c: 83 f8 81                    	cmpl	$-127, %eax
100005b6f: 0f 8f 7b ff ff ff           	jg	-133 <__ZN11LineNetwork7forwardEv+0x420>
100005b75: b8 81 00 00 00              	movl	$129, %eax
100005b7a: e9 71 ff ff ff              	jmp	-143 <__ZN11LineNetwork7forwardEv+0x420>
100005b7f: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100005b83: 48 89 df                    	movq	%rbx, %rdi
100005b86: e8 35 f0 ff ff              	callq	-4043 <__ZN14ModelInterface11swap_bufferEv>
100005b8b: 48 89 df                    	movq	%rbx, %rdi
100005b8e: e8 7d ee ff ff              	callq	-4483 <__ZN14ModelInterface13output_bufferEv>
100005b93: 49 89 c6                    	movq	%rax, %r14
100005b96: 48 89 df                    	movq	%rbx, %rdi
100005b99: e8 62 ee ff ff              	callq	-4510 <__ZN14ModelInterface12input_bufferEv>
100005b9e: 4c 89 f7                    	movq	%r14, %rdi
100005ba1: 48 89 c6                    	movq	%rax, %rsi
100005ba4: ba 00 80 00 00              	movl	$32768, %edx
100005ba9: e8 22 f0 ff ff              	callq	-4062 <__Z4ReLUPaS_j>
100005bae: 48 89 df                    	movq	%rbx, %rdi
100005bb1: e8 0a f0 ff ff              	callq	-4086 <__ZN14ModelInterface11swap_bufferEv>
100005bb6: 48 89 df                    	movq	%rbx, %rdi
100005bb9: e8 52 ee ff ff              	callq	-4526 <__ZN14ModelInterface13output_bufferEv>
100005bbe: 49 89 c4                    	movq	%rax, %r12
100005bc1: 48 89 df                    	movq	%rbx, %rdi
100005bc4: e8 37 ee ff ff              	callq	-4553 <__ZN14ModelInterface12input_bufferEv>
100005bc9: 49 89 c6                    	movq	%rax, %r14
100005bcc: 31 c0                       	xorl	%eax, %eax
100005bce: 4c 8d 3d db 30 00 00        	leaq	12507(%rip), %r15
100005bd5: eb 21                       	jmp	33 <__ZN11LineNetwork7forwardEv+0x528>
100005bd7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100005be0: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100005be4: 48 ff c0                    	incq	%rax
100005be7: 49 83 c4 20                 	addq	$32, %r12
100005beb: 49 81 c6 00 04 00 00        	addq	$1024, %r14
100005bf2: 48 83 f8 20                 	cmpq	$32, %rax
100005bf6: 74 63                       	je	99 <__ZN11LineNetwork7forwardEv+0x58b>
100005bf8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100005bfc: 4c 89 f3                    	movq	%r14, %rbx
100005bff: 45 31 ed                    	xorl	%r13d, %r13d
100005c02: eb 1d                       	jmp	29 <__ZN11LineNetwork7forwardEv+0x551>
100005c04: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005c0e: 66 90                       	nop
100005c10: 43 88 04 2c                 	movb	%al, (%r12,%r13)
100005c14: 49 ff c5                    	incq	%r13
100005c17: 48 83 c3 20                 	addq	$32, %rbx
100005c1b: 49 83 fd 20                 	cmpq	$32, %r13
100005c1f: 74 bf                       	je	-65 <__ZN11LineNetwork7forwardEv+0x510>
100005c21: 48 89 df                    	movq	%rbx, %rdi
100005c24: 4c 89 fe                    	movq	%r15, %rsi
100005c27: e8 44 11 00 00              	callq	4420 <__ZN11LineNetwork7forwardEv+0x16a0>
100005c2c: c1 e0 05                    	shll	$5, %eax
100005c2f: 89 c1                       	movl	%eax, %ecx
100005c31: 83 c1 20                    	addl	$32, %ecx
100005c34: c1 f9 1f                    	sarl	$31, %ecx
100005c37: c1 e9 12                    	shrl	$18, %ecx
100005c3a: 8d 04 08                    	leal	(%rax,%rcx), %eax
100005c3d: 83 c0 20                    	addl	$32, %eax
100005c40: c1 f8 0e                    	sarl	$14, %eax
100005c43: 3d 80 00 00 00              	cmpl	$128, %eax
100005c48: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x57f>
100005c4a: b8 7f 00 00 00              	movl	$127, %eax
100005c4f: 83 f8 81                    	cmpl	$-127, %eax
100005c52: 7f bc                       	jg	-68 <__ZN11LineNetwork7forwardEv+0x540>
100005c54: b8 81 00 00 00              	movl	$129, %eax
100005c59: eb b5                       	jmp	-75 <__ZN11LineNetwork7forwardEv+0x540>
100005c5b: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100005c5f: 48 89 df                    	movq	%rbx, %rdi
100005c62: e8 59 ef ff ff              	callq	-4263 <__ZN14ModelInterface11swap_bufferEv>
100005c67: 48 89 df                    	movq	%rbx, %rdi
100005c6a: 48 83 c4 58                 	addq	$88, %rsp
100005c6e: 5b                          	popq	%rbx
100005c6f: 41 5c                       	popq	%r12
100005c71: 41 5d                       	popq	%r13
100005c73: 41 5e                       	popq	%r14
100005c75: 41 5f                       	popq	%r15
100005c77: 5d                          	popq	%rbp
100005c78: e9 43 ef ff ff              	jmp	-4285 <__ZN14ModelInterface11swap_bufferEv>
100005c7d: 0f 1f 00                    	nopl	(%rax)
100005c80: 55                          	pushq	%rbp
100005c81: 48 89 e5                    	movq	%rsp, %rbp
100005c84: 41 57                       	pushq	%r15
100005c86: 41 56                       	pushq	%r14
100005c88: 41 55                       	pushq	%r13
100005c8a: 41 54                       	pushq	%r12
100005c8c: 53                          	pushq	%rbx
100005c8d: 48 83 e4 e0                 	andq	$-32, %rsp
100005c91: 48 81 ec e0 02 00 00        	subq	$736, %rsp
100005c98: 48 89 4c 24 50              	movq	%rcx, 80(%rsp)
100005c9d: 48 89 54 24 48              	movq	%rdx, 72(%rsp)
100005ca2: 49 89 ff                    	movq	%rdi, %r15
100005ca5: c4 c1 79 6e c0              	vmovd	%r8d, %xmm0
100005caa: c4 e2 7d 58 c8              	vpbroadcastd	%xmm0, %ymm1
100005caf: 48 8d 86 01 04 00 00        	leaq	1025(%rsi), %rax
100005cb6: 48 89 44 24 40              	movq	%rax, 64(%rsp)
100005cbb: 48 8d 86 02 04 00 00        	leaq	1026(%rsi), %rax
100005cc2: 48 89 44 24 38              	movq	%rax, 56(%rsp)
100005cc7: 45 31 c9                    	xorl	%r9d, %r9d
100005cca: c5 fd 6f 15 0e 16 00 00     	vmovdqa	5646(%rip), %ymm2
100005cd2: 44 89 44 24 14              	movl	%r8d, 20(%rsp)
100005cd7: 48 89 74 24 58              	movq	%rsi, 88(%rsp)
100005cdc: c5 fd 7f 8c 24 60 02 00 00  	vmovdqa	%ymm1, 608(%rsp)
100005ce5: eb 19                       	jmp	25 <__ZN11LineNetwork7forwardEv+0x630>
100005ce7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100005cf0: 49 ff c1                    	incq	%r9
100005cf3: 48 ff c7                    	incq	%rdi
100005cf6: 49 83 f9 08                 	cmpq	$8, %r9
100005cfa: 0f 84 f2 0c 00 00           	je	3314 <__ZN11LineNetwork7forwardEv+0x1322>
100005d00: 49 8d 81 f1 07 00 00        	leaq	2033(%r9), %rax
100005d07: 48 89 84 24 88 00 00 00     	movq	%rax, 136(%rsp)
100005d0f: 4b 8d 04 c9                 	leaq	(%r9,%r9,8), %rax
100005d13: 48 8b 54 24 48              	movq	72(%rsp), %rdx
100005d18: 48 8d 0c 02                 	leaq	(%rdx,%rax), %rcx
100005d1c: 48 83 c1 09                 	addq	$9, %rcx
100005d20: 48 89 8c 24 80 00 00 00     	movq	%rcx, 128(%rsp)
100005d28: 48 8b 4c 24 50              	movq	80(%rsp), %rcx
100005d2d: 48 8d 5c 01 01              	leaq	1(%rcx,%rax), %rbx
100005d32: 48 89 5c 24 78              	movq	%rbx, 120(%rsp)
100005d37: 4c 8d 14 02                 	leaq	(%rdx,%rax), %r10
100005d3b: 4c 8d 1c 01                 	leaq	(%rcx,%rax), %r11
100005d3f: 48 8d 44 02 08              	leaq	8(%rdx,%rax), %rax
100005d44: 48 89 44 24 70              	movq	%rax, 112(%rsp)
100005d49: c4 c1 f9 6e c1              	vmovq	%r9, %xmm0
100005d4e: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005d53: 41 be 00 00 00 00           	movl	$0, %r14d
100005d59: 48 8b 44 24 38              	movq	56(%rsp), %rax
100005d5e: 48 89 44 24 30              	movq	%rax, 48(%rsp)
100005d63: 48 8b 44 24 40              	movq	64(%rsp), %rax
100005d68: 31 c9                       	xorl	%ecx, %ecx
100005d6a: 31 d2                       	xorl	%edx, %edx
100005d6c: 48 89 54 24 08              	movq	%rdx, 8(%rsp)
100005d71: 4c 89 4c 24 68              	movq	%r9, 104(%rsp)
100005d76: 48 89 7c 24 60              	movq	%rdi, 96(%rsp)
100005d7b: 4c 89 54 24 20              	movq	%r10, 32(%rsp)
100005d80: 4c 89 5c 24 18              	movq	%r11, 24(%rsp)
100005d85: c5 fd 7f 84 24 80 02 00 00  	vmovdqa	%ymm0, 640(%rsp)
100005d8e: eb 38                       	jmp	56 <__ZN11LineNetwork7forwardEv+0x6f8>
100005d90: 48 8b 8c 24 90 00 00 00     	movq	144(%rsp), %rcx
100005d98: 48 ff c1                    	incq	%rcx
100005d9b: 48 8b 44 24 28              	movq	40(%rsp), %rax
100005da0: 48 05 00 04 00 00           	addq	$1024, %rax
100005da6: 48 81 44 24 30 00 04 00 00  	addq	$1024, 48(%rsp)
100005daf: 49 81 c6 00 01 00 00        	addq	$256, %r14
100005db6: 48 81 7c 24 08 fd 01 00 00  	cmpq	$509, 8(%rsp)
100005dbf: 4d 89 e9                    	movq	%r13, %r9
100005dc2: 0f 83 28 ff ff ff           	jae	-216 <__ZN11LineNetwork7forwardEv+0x620>
100005dc8: 48 89 44 24 28              	movq	%rax, 40(%rsp)
100005dcd: 4c 89 b4 24 98 00 00 00     	movq	%r14, 152(%rsp)
100005dd5: 48 89 cb                    	movq	%rcx, %rbx
100005dd8: 48 c1 e3 0b                 	shlq	$11, %rbx
100005ddc: 4d 89 cd                    	movq	%r9, %r13
100005ddf: 49 8d 04 19                 	leaq	(%r9,%rbx), %rax
100005de3: 4c 01 f8                    	addq	%r15, %rax
100005de6: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100005dee: 4c 01 fb                    	addq	%r15, %rbx
100005df1: 48 89 ca                    	movq	%rcx, %rdx
100005df4: 48 c1 e2 0a                 	shlq	$10, %rdx
100005df8: 4c 8d 0c 16                 	leaq	(%rsi,%rdx), %r9
100005dfc: 49 81 c1 ff 05 00 00        	addq	$1535, %r9
100005e03: 48 01 f2                    	addq	%rsi, %rdx
100005e06: 4c 39 c8                    	cmpq	%r9, %rax
100005e09: 41 0f 92 c4                 	setb	%r12b
100005e0d: 48 39 da                    	cmpq	%rbx, %rdx
100005e10: 41 0f 92 c2                 	setb	%r10b
100005e14: 48 3b 84 24 80 00 00 00     	cmpq	128(%rsp), %rax
100005e1c: 41 0f 92 c6                 	setb	%r14b
100005e20: 48 39 5c 24 70              	cmpq	%rbx, 112(%rsp)
100005e25: 4c 89 da                    	movq	%r11, %rdx
100005e28: 41 0f 92 c3                 	setb	%r11b
100005e2c: 48 3b 44 24 78              	cmpq	120(%rsp), %rax
100005e31: 0f 92 c0                    	setb	%al
100005e34: 48 39 da                    	cmpq	%rbx, %rdx
100005e37: 41 0f 92 c1                 	setb	%r9b
100005e3b: 45 84 d4                    	testb	%r10b, %r12b
100005e3e: 48 89 8c 24 90 00 00 00     	movq	%rcx, 144(%rsp)
100005e46: 0f 85 84 0a 00 00           	jne	2692 <__ZN11LineNetwork7forwardEv+0x1200>
100005e4c: 45 20 de                    	andb	%r11b, %r14b
100005e4f: 0f 85 7b 0a 00 00           	jne	2683 <__ZN11LineNetwork7forwardEv+0x1200>
100005e55: ba 00 00 00 00              	movl	$0, %edx
100005e5a: 44 20 c8                    	andb	%r9b, %al
100005e5d: 0f 85 6f 0a 00 00           	jne	2671 <__ZN11LineNetwork7forwardEv+0x1202>
100005e63: 48 8b 44 24 08              	movq	8(%rsp), %rax
100005e68: 48 c1 e0 07                 	shlq	$7, %rax
100005e6c: c4 e1 f9 6e c0              	vmovq	%rax, %xmm0
100005e71: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005e76: c5 fd 7f 84 24 a0 02 00 00  	vmovdqa	%ymm0, 672(%rsp)
100005e7f: 45 31 db                    	xorl	%r11d, %r11d
100005e82: c5 fc 28 05 36 14 00 00     	vmovaps	5174(%rip), %ymm0
100005e8a: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100005e93: c5 fc 28 05 05 14 00 00     	vmovaps	5125(%rip), %ymm0
100005e9b: c5 fc 29 84 24 20 02 00 00  	vmovaps	%ymm0, 544(%rsp)
100005ea4: c5 fc 28 05 d4 13 00 00     	vmovaps	5076(%rip), %ymm0
100005eac: c5 fc 29 84 24 00 02 00 00  	vmovaps	%ymm0, 512(%rsp)
100005eb5: c5 fc 28 05 a3 13 00 00     	vmovaps	5027(%rip), %ymm0
100005ebd: c5 fc 29 84 24 e0 01 00 00  	vmovaps	%ymm0, 480(%rsp)
100005ec6: c5 fc 28 05 72 13 00 00     	vmovaps	4978(%rip), %ymm0
100005ece: c5 fc 29 84 24 c0 01 00 00  	vmovaps	%ymm0, 448(%rsp)
100005ed7: c5 fc 28 05 41 13 00 00     	vmovaps	4929(%rip), %ymm0
100005edf: c5 fc 29 84 24 a0 01 00 00  	vmovaps	%ymm0, 416(%rsp)
100005ee8: c5 fc 28 05 10 13 00 00     	vmovaps	4880(%rip), %ymm0
100005ef0: c5 fc 29 84 24 80 01 00 00  	vmovaps	%ymm0, 384(%rsp)
100005ef9: c5 fc 28 05 df 12 00 00     	vmovaps	4831(%rip), %ymm0
100005f01: c5 fc 29 84 24 60 01 00 00  	vmovaps	%ymm0, 352(%rsp)
100005f0a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100005f10: 48 8b 4c 24 28              	movq	40(%rsp), %rcx
100005f15: c4 a1 7e 6f 84 59 1f fc ff ff       	vmovdqu	-993(%rcx,%r11,2), %ymm0
100005f1f: c4 e2 7d 00 c2              	vpshufb	%ymm2, %ymm0, %ymm0
100005f24: c4 a1 7e 6f 8c 59 ff fb ff ff       	vmovdqu	-1025(%rcx,%r11,2), %ymm1
100005f2e: c4 21 7e 6f 84 59 00 fc ff ff       	vmovdqu	-1024(%rcx,%r11,2), %ymm8
100005f38: c5 7d 6f 1d c0 13 00 00     	vmovdqa	5056(%rip), %ymm11
100005f40: c4 c2 75 00 cb              	vpshufb	%ymm11, %ymm1, %ymm1
100005f45: c4 e3 75 02 c0 cc           	vpblendd	$204, %ymm0, %ymm1, %ymm0
100005f4b: c4 e3 fd 00 c8 d8           	vpermq	$216, %ymm0, %ymm1
100005f51: c4 a1 7a 6f 94 59 0f fc ff ff       	vmovdqu	-1009(%rcx,%r11,2), %xmm2
100005f5b: c5 f9 6f 1d 4d 12 00 00     	vmovdqa	4685(%rip), %xmm3
100005f63: c4 e2 69 00 d3              	vpshufb	%xmm3, %xmm2, %xmm2
100005f68: c5 79 6f e3                 	vmovdqa	%xmm3, %xmm12
100005f6c: c4 62 7d 21 ca              	vpmovsxbd	%xmm2, %ymm9
100005f71: c4 63 fd 00 d0 db           	vpermq	$219, %ymm0, %ymm10
100005f77: 48 8b 44 24 20              	movq	32(%rsp), %rax
100005f7c: c4 e2 79 78 00              	vpbroadcastb	(%rax), %xmm0
100005f81: c4 e2 7d 21 d0              	vpmovsxbd	%xmm0, %ymm2
100005f86: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
100005f8b: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100005f94: c4 62 7d 21 c9              	vpmovsxbd	%xmm1, %ymm9
100005f99: c4 42 7d 21 d2              	vpmovsxbd	%xmm10, %ymm10
100005f9e: c4 21 7e 6f ac 59 20 fc ff ff       	vmovdqu	-992(%rcx,%r11,2), %ymm13
100005fa8: c4 62 15 00 3d 2f 13 00 00  	vpshufb	4911(%rip), %ymm13, %ymm15
100005fb1: c4 c2 3d 00 fb              	vpshufb	%ymm11, %ymm8, %ymm7
100005fb6: c4 c3 45 02 ff cc           	vpblendd	$204, %ymm15, %ymm7, %ymm7
100005fbc: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
100005fc2: c4 63 fd 00 ff d8           	vpermq	$216, %ymm7, %ymm15
100005fc8: c5 fd 6f 05 50 13 00 00     	vmovdqa	4944(%rip), %ymm0
100005fd0: c4 62 15 00 e8              	vpshufb	%ymm0, %ymm13, %ymm13
100005fd5: c5 fd 6f 05 63 13 00 00     	vmovdqa	4963(%rip), %ymm0
100005fdd: c4 62 3d 00 c0              	vpshufb	%ymm0, %ymm8, %ymm8
100005fe2: c4 c3 3d 02 f5 cc           	vpblendd	$204, %ymm13, %ymm8, %ymm6
100005fe8: c4 e3 fd 00 ee d8           	vpermq	$216, %ymm6, %ymm5
100005fee: c4 e2 7d 21 e1              	vpmovsxbd	%xmm1, %ymm4
100005ff3: c4 c2 7d 21 df              	vpmovsxbd	%xmm15, %ymm3
100005ff8: c4 e3 fd 00 cf db           	vpermq	$219, %ymm7, %ymm1
100005ffe: c4 62 7d 21 e9              	vpmovsxbd	%xmm1, %ymm13
100006003: c4 43 7d 39 ff 01           	vextracti128	$1, %ymm15, %xmm15
100006009: c4 42 6d 40 c2              	vpmulld	%ymm10, %ymm2, %ymm8
10000600e: c4 21 7a 6f 94 59 10 fc ff ff       	vmovdqu	-1008(%rcx,%r11,2), %xmm10
100006018: c4 c2 29 00 fc              	vpshufb	%xmm12, %xmm10, %xmm7
10000601d: c4 62 79 78 70 01           	vpbroadcastb	1(%rax), %xmm14
100006023: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
100006028: c5 fd 7f 84 24 a0 00 00 00  	vmovdqa	%ymm0, 160(%rsp)
100006031: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
100006036: c4 42 7d 21 f6              	vpmovsxbd	%xmm14, %ymm14
10000603b: c4 e2 0d 40 ff              	vpmulld	%ymm7, %ymm14, %ymm7
100006040: c4 42 0d 40 ed              	vpmulld	%ymm13, %ymm14, %ymm13
100006045: c4 42 7d 21 e7              	vpmovsxbd	%xmm15, %ymm12
10000604a: c4 c3 7d 39 ef 01           	vextracti128	$1, %ymm5, %xmm15
100006050: c4 e3 fd 00 f6 db           	vpermq	$219, %ymm6, %ymm6
100006056: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
10000605b: c4 62 0d 40 cb              	vpmulld	%ymm3, %ymm14, %ymm9
100006060: c4 e2 7d 21 dd              	vpmovsxbd	%xmm5, %ymm3
100006065: c5 f9 6f 05 53 11 00 00     	vmovdqa	4435(%rip), %xmm0
10000606d: c4 e2 29 00 e8              	vpshufb	%xmm0, %xmm10, %xmm5
100006072: c4 e2 79 78 40 02           	vpbroadcastb	2(%rax), %xmm0
100006078: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
10000607d: c4 c2 7d 21 cf              	vpmovsxbd	%xmm15, %ymm1
100006082: c4 62 7d 40 fb              	vpmulld	%ymm3, %ymm0, %ymm15
100006087: c4 62 7d 40 d6              	vpmulld	%ymm6, %ymm0, %ymm10
10000608c: c4 e2 6d 40 d4              	vpmulld	%ymm4, %ymm2, %ymm2
100006091: c5 fd 7f 94 24 40 01 00 00  	vmovdqa	%ymm2, 320(%rsp)
10000609a: c4 e2 7d 21 d5              	vpmovsxbd	%xmm5, %ymm2
10000609f: c4 e2 7d 40 d2              	vpmulld	%ymm2, %ymm0, %ymm2
1000060a4: c4 a1 7e 6f 9c 59 ff fd ff ff       	vmovdqu	-513(%rcx,%r11,2), %ymm3
1000060ae: c4 c2 0d 40 e4              	vpmulld	%ymm12, %ymm14, %ymm4
1000060b3: c5 fd 7f a4 24 00 01 00 00  	vmovdqa	%ymm4, 256(%rsp)
1000060bc: c4 a1 7e 6f a4 59 1f fe ff ff       	vmovdqu	-481(%rcx,%r11,2), %ymm4
1000060c6: c4 e2 5d 00 25 11 12 00 00  	vpshufb	4625(%rip), %ymm4, %ymm4
1000060cf: c4 c2 65 00 db              	vpshufb	%ymm11, %ymm3, %ymm3
1000060d4: c4 e3 65 02 dc cc           	vpblendd	$204, %ymm4, %ymm3, %ymm3
1000060da: c4 e2 7d 40 c1              	vpmulld	%ymm1, %ymm0, %ymm0
1000060df: c5 fd 7f 84 24 20 01 00 00  	vmovdqa	%ymm0, 288(%rsp)
1000060e8: c4 e3 fd 00 c3 d8           	vpermq	$216, %ymm3, %ymm0
1000060ee: c4 e2 7d 21 c8              	vpmovsxbd	%xmm0, %ymm1
1000060f3: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
1000060f9: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000060fe: c4 e3 7d 39 c0 01           	vextracti128	$1, %ymm0, %xmm0
100006104: c5 c5 fe a4 24 c0 00 00 00  	vpaddd	192(%rsp), %ymm7, %ymm4
10000610d: c4 a1 7a 6f ac 59 0f fe ff ff       	vmovdqu	-497(%rcx,%r11,2), %xmm5
100006117: c5 79 6f 35 91 10 00 00     	vmovdqa	4241(%rip), %xmm14
10000611f: c4 c2 51 00 ee              	vpshufb	%xmm14, %xmm5, %xmm5
100006124: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006129: c4 e2 79 78 70 03           	vpbroadcastb	3(%rax), %xmm6
10000612f: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006134: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
100006139: c4 e2 4d 40 c0              	vpmulld	%ymm0, %ymm6, %ymm0
10000613e: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100006147: c4 e2 4d 40 db              	vpmulld	%ymm3, %ymm6, %ymm3
10000614c: c4 41 3d fe ed              	vpaddd	%ymm13, %ymm8, %ymm13
100006151: c4 e2 4d 40 c9              	vpmulld	%ymm1, %ymm6, %ymm1
100006156: c4 e2 4d 40 c5              	vpmulld	%ymm5, %ymm6, %ymm0
10000615b: c5 ed fe c0                 	vpaddd	%ymm0, %ymm2, %ymm0
10000615f: c5 35 fe 84 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm9, %ymm8
100006168: c5 dd fe c0                 	vpaddd	%ymm0, %ymm4, %ymm0
10000616c: c5 fd 7f 84 24 e0 00 00 00  	vmovdqa	%ymm0, 224(%rsp)
100006175: c4 a1 7e 6f a4 59 00 fe ff ff       	vmovdqu	-512(%rcx,%r11,2), %ymm4
10000617f: c4 a1 7e 6f ac 59 20 fe ff ff       	vmovdqu	-480(%rcx,%r11,2), %ymm5
100006189: c4 e2 55 00 35 4e 11 00 00  	vpshufb	4430(%rip), %ymm5, %ymm6
100006192: c4 c2 5d 00 fb              	vpshufb	%ymm11, %ymm4, %ymm7
100006197: c5 2d fe d3                 	vpaddd	%ymm3, %ymm10, %ymm10
10000619b: c4 e3 45 02 de cc           	vpblendd	$204, %ymm6, %ymm7, %ymm3
1000061a1: c4 e3 fd 00 f3 d8           	vpermq	$216, %ymm3, %ymm6
1000061a7: c4 e3 7d 39 f7 01           	vextracti128	$1, %ymm6, %xmm7
1000061ad: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
1000061b2: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
1000061b8: c5 05 fe c9                 	vpaddd	%ymm1, %ymm15, %ymm9
1000061bc: c4 e2 7d 21 cb              	vpmovsxbd	%xmm3, %ymm1
1000061c1: c4 e2 7d 21 de              	vpmovsxbd	%xmm6, %ymm3
1000061c6: c4 a1 7a 6f b4 59 10 fe ff ff       	vmovdqu	-496(%rcx,%r11,2), %xmm6
1000061d0: c4 e2 79 78 40 04           	vpbroadcastb	4(%rax), %xmm0
1000061d6: c4 c2 49 00 d6              	vpshufb	%xmm14, %xmm6, %xmm2
1000061db: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000061e0: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
1000061e5: c4 e2 7d 40 db              	vpmulld	%ymm3, %ymm0, %ymm3
1000061ea: c4 62 7d 40 e1              	vpmulld	%ymm1, %ymm0, %ymm12
1000061ef: c4 e2 7d 40 cf              	vpmulld	%ymm7, %ymm0, %ymm1
1000061f4: c5 fd 7f 8c 24 a0 00 00 00  	vmovdqa	%ymm1, 160(%rsp)
1000061fd: c4 e2 7d 40 c2              	vpmulld	%ymm2, %ymm0, %ymm0
100006202: c4 e2 55 00 15 15 11 00 00  	vpshufb	4373(%rip), %ymm5, %ymm2
10000620b: c4 e2 5d 00 25 2c 11 00 00  	vpshufb	4396(%rip), %ymm4, %ymm4
100006214: c4 e3 5d 02 d2 cc           	vpblendd	$204, %ymm2, %ymm4, %ymm2
10000621a: c4 e2 79 78 60 05           	vpbroadcastb	5(%rax), %xmm4
100006220: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100006225: c4 e3 fd 00 ea db           	vpermq	$219, %ymm2, %ymm5
10000622b: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006230: c4 e2 5d 40 ed              	vpmulld	%ymm5, %ymm4, %ymm5
100006235: c5 9d fe ed                 	vpaddd	%ymm5, %ymm12, %ymm5
100006239: c4 e3 fd 00 d2 d8           	vpermq	$216, %ymm2, %ymm2
10000623f: c4 e2 7d 21 fa              	vpmovsxbd	%xmm2, %ymm7
100006244: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000624a: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000624f: c4 e2 49 00 35 68 0f 00 00  	vpshufb	3944(%rip), %xmm6, %xmm6
100006258: c4 e2 5d 40 ff              	vpmulld	%ymm7, %ymm4, %ymm7
10000625d: c4 62 5d 40 fa              	vpmulld	%ymm2, %ymm4, %ymm15
100006262: c4 e2 7d 21 d6              	vpmovsxbd	%xmm6, %ymm2
100006267: c4 e2 5d 40 d2              	vpmulld	%ymm2, %ymm4, %ymm2
10000626c: c5 fd fe c2                 	vpaddd	%ymm2, %ymm0, %ymm0
100006270: c4 a1 7e 6f 54 59 ff        	vmovdqu	-1(%rcx,%r11,2), %ymm2
100006277: c5 e5 fe df                 	vpaddd	%ymm7, %ymm3, %ymm3
10000627b: c4 a1 7e 6f 64 59 1f        	vmovdqu	31(%rcx,%r11,2), %ymm4
100006282: c4 e2 5d 00 25 55 10 00 00  	vpshufb	4181(%rip), %ymm4, %ymm4
10000628b: c4 c2 6d 00 d3              	vpshufb	%ymm11, %ymm2, %ymm2
100006290: c4 e3 6d 02 d4 cc           	vpblendd	$204, %ymm4, %ymm2, %ymm2
100006296: c4 e3 fd 00 e2 d8           	vpermq	$216, %ymm2, %ymm4
10000629c: c4 c1 15 fe f2              	vpaddd	%ymm10, %ymm13, %ymm6
1000062a1: c4 e3 7d 39 e7 01           	vextracti128	$1, %ymm4, %xmm7
1000062a7: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
1000062ac: c4 e3 fd 00 d2 db           	vpermq	$219, %ymm2, %ymm2
1000062b2: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000062b7: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000062bc: c4 41 3d fe c1              	vpaddd	%ymm9, %ymm8, %ymm8
1000062c1: c4 e2 79 78 48 06           	vpbroadcastb	6(%rax), %xmm1
1000062c7: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000062cc: c4 e2 75 40 e4              	vpmulld	%ymm4, %ymm1, %ymm4
1000062d1: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
1000062d5: c4 a1 7a 6f 64 59 0f        	vmovdqu	15(%rcx,%r11,2), %xmm4
1000062dc: c4 c2 59 00 e6              	vpshufb	%xmm14, %xmm4, %xmm4
1000062e1: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000062e6: c4 e2 75 40 d2              	vpmulld	%ymm2, %ymm1, %ymm2
1000062eb: c5 d5 fe d2                 	vpaddd	%ymm2, %ymm5, %ymm2
1000062ef: c4 62 75 40 ef              	vpmulld	%ymm7, %ymm1, %ymm13
1000062f4: c4 e2 75 40 cc              	vpmulld	%ymm4, %ymm1, %ymm1
1000062f9: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
1000062fd: c5 3d fe c3                 	vpaddd	%ymm3, %ymm8, %ymm8
100006301: c5 7d fe 94 24 e0 00 00 00  	vpaddd	224(%rsp), %ymm0, %ymm10
10000630a: c4 a1 7e 6f 0c 59           	vmovdqu	(%rcx,%r11,2), %ymm1
100006310: c4 a1 7e 6f 5c 59 20        	vmovdqu	32(%rcx,%r11,2), %ymm3
100006317: c4 e2 65 00 25 c0 0f 00 00  	vpshufb	4032(%rip), %ymm3, %ymm4
100006320: c4 c2 75 00 eb              	vpshufb	%ymm11, %ymm1, %ymm5
100006325: c5 4d fe da                 	vpaddd	%ymm2, %ymm6, %ymm11
100006329: c4 e3 55 02 e4 cc           	vpblendd	$204, %ymm4, %ymm5, %ymm4
10000632f: c4 e3 fd 00 ec d8           	vpermq	$216, %ymm4, %ymm5
100006335: c4 e2 65 00 1d e2 0f 00 00  	vpshufb	4066(%rip), %ymm3, %ymm3
10000633e: c4 e2 75 00 0d f9 0f 00 00  	vpshufb	4089(%rip), %ymm1, %ymm1
100006347: c4 e3 75 02 cb cc           	vpblendd	$204, %ymm3, %ymm1, %ymm1
10000634d: c5 fd 6f 84 24 00 01 00 00  	vmovdqa	256(%rsp), %ymm0
100006356: c5 7d fe a4 24 40 01 00 00  	vpaddd	320(%rsp), %ymm0, %ymm12
10000635f: c4 e3 fd 00 f1 d8           	vpermq	$216, %ymm1, %ymm6
100006365: c4 e2 7d 21 fd              	vpmovsxbd	%xmm5, %ymm7
10000636a: c4 e3 fd 00 e4 db           	vpermq	$219, %ymm4, %ymm4
100006370: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100006375: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
10000637b: c5 fd 6f 84 24 c0 00 00 00  	vmovdqa	192(%rsp), %ymm0
100006384: c5 7d fe 8c 24 20 01 00 00  	vpaddd	288(%rsp), %ymm0, %ymm9
10000638d: c4 a1 7a 6f 44 59 10        	vmovdqu	16(%rcx,%r11,2), %xmm0
100006394: c4 c2 79 00 d6              	vpshufb	%xmm14, %xmm0, %xmm2
100006399: c4 e2 79 78 58 07           	vpbroadcastb	7(%rax), %xmm3
10000639f: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000063a4: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
1000063a9: c4 e2 65 40 e4              	vpmulld	%ymm4, %ymm3, %ymm4
1000063ae: c4 e2 65 40 ff              	vpmulld	%ymm7, %ymm3, %ymm7
1000063b3: c4 e2 65 40 ed              	vpmulld	%ymm5, %ymm3, %ymm5
1000063b8: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000063bd: c4 e2 65 40 d2              	vpmulld	%ymm2, %ymm3, %ymm2
1000063c2: c4 e2 79 78 58 08           	vpbroadcastb	8(%rax), %xmm3
1000063c8: c5 05 fe b4 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm15, %ymm14
1000063d1: c4 62 7d 21 fe              	vpmovsxbd	%xmm6, %ymm15
1000063d6: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000063db: c4 42 65 40 ff              	vpmulld	%ymm15, %ymm3, %ymm15
1000063e0: c4 c1 45 fe ff              	vpaddd	%ymm15, %ymm7, %ymm7
1000063e5: c4 41 1d fe c9              	vpaddd	%ymm9, %ymm12, %ymm9
1000063ea: c4 e3 fd 00 c9 db           	vpermq	$219, %ymm1, %ymm1
1000063f0: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000063f5: c4 e2 65 40 c9              	vpmulld	%ymm1, %ymm3, %ymm1
1000063fa: c5 dd fe c9                 	vpaddd	%ymm1, %ymm4, %ymm1
1000063fe: c4 c1 0d fe e5              	vpaddd	%ymm13, %ymm14, %ymm4
100006403: c4 e3 7d 39 f6 01           	vextracti128	$1, %ymm6, %xmm6
100006409: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
10000640e: c4 e2 65 40 f6              	vpmulld	%ymm6, %ymm3, %ymm6
100006413: c5 d5 fe ee                 	vpaddd	%ymm6, %ymm5, %ymm5
100006417: c4 e2 79 00 05 a0 0d 00 00  	vpshufb	3488(%rip), %xmm0, %xmm0
100006420: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006425: c4 e2 65 40 c0              	vpmulld	%ymm0, %ymm3, %ymm0
10000642a: c5 ed fe c0                 	vpaddd	%ymm0, %ymm2, %ymm0
10000642e: 48 8b 44 24 18              	movq	24(%rsp), %rax
100006433: c4 e2 79 78 10              	vpbroadcastb	(%rax), %xmm2
100006438: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000643d: c5 c5 fe da                 	vpaddd	%ymm2, %ymm7, %ymm3
100006441: c5 bd fe db                 	vpaddd	%ymm3, %ymm8, %ymm3
100006445: c5 f5 fe ca                 	vpaddd	%ymm2, %ymm1, %ymm1
100006449: c5 a5 fe c9                 	vpaddd	%ymm1, %ymm11, %ymm1
10000644d: c5 b5 fe e4                 	vpaddd	%ymm4, %ymm9, %ymm4
100006451: c5 d5 fe ea                 	vpaddd	%ymm2, %ymm5, %ymm5
100006455: c5 fd fe c2                 	vpaddd	%ymm2, %ymm0, %ymm0
100006459: c5 ad fe c0                 	vpaddd	%ymm0, %ymm10, %ymm0
10000645d: c5 fd 6f b4 24 60 02 00 00  	vmovdqa	608(%rsp), %ymm6
100006466: c4 e2 75 40 ce              	vpmulld	%ymm6, %ymm1, %ymm1
10000646b: c5 dd fe d5                 	vpaddd	%ymm5, %ymm4, %ymm2
10000646f: c4 e2 65 40 de              	vpmulld	%ymm6, %ymm3, %ymm3
100006474: c4 e2 7d 40 c6              	vpmulld	%ymm6, %ymm0, %ymm0
100006479: c4 e2 6d 40 d6              	vpmulld	%ymm6, %ymm2, %ymm2
10000647e: c5 dd 72 e3 1f              	vpsrad	$31, %ymm3, %ymm4
100006483: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
100006488: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
10000648c: c5 dd 72 e1 1f              	vpsrad	$31, %ymm1, %ymm4
100006491: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
100006496: c5 e5 72 e3 0e              	vpsrad	$14, %ymm3, %ymm3
10000649b: c5 f5 fe cc                 	vpaddd	%ymm4, %ymm1, %ymm1
10000649f: c5 f5 72 e1 0e              	vpsrad	$14, %ymm1, %ymm1
1000064a4: c5 dd 72 e2 1f              	vpsrad	$31, %ymm2, %ymm4
1000064a9: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
1000064ae: c5 ed fe d4                 	vpaddd	%ymm4, %ymm2, %ymm2
1000064b2: c5 ed 72 e2 0e              	vpsrad	$14, %ymm2, %ymm2
1000064b7: c5 dd 72 e0 1f              	vpsrad	$31, %ymm0, %ymm4
1000064bc: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
1000064c1: c5 fd fe c4                 	vpaddd	%ymm4, %ymm0, %ymm0
1000064c5: c5 fd 72 e0 0e              	vpsrad	$14, %ymm0, %ymm0
1000064ca: c4 e2 7d 58 25 0d 28 00 00  	vpbroadcastd	10253(%rip), %ymm4
1000064d3: c4 e2 6d 39 d4              	vpminsd	%ymm4, %ymm2, %ymm2
1000064d8: c4 e2 75 39 cc              	vpminsd	%ymm4, %ymm1, %ymm1
1000064dd: c4 e2 65 39 dc              	vpminsd	%ymm4, %ymm3, %ymm3
1000064e2: c4 e2 7d 39 e4              	vpminsd	%ymm4, %ymm0, %ymm4
1000064e7: c4 e2 7d 58 2d f4 27 00 00  	vpbroadcastd	10228(%rip), %ymm5
1000064f0: c4 e2 75 3d c5              	vpmaxsd	%ymm5, %ymm1, %ymm0
1000064f5: c4 e2 6d 3d cd              	vpmaxsd	%ymm5, %ymm2, %ymm1
1000064fa: c5 f5 6b c0                 	vpackssdw	%ymm0, %ymm1, %ymm0
1000064fe: c4 e2 65 3d cd              	vpmaxsd	%ymm5, %ymm3, %ymm1
100006503: c4 e2 5d 3d d5              	vpmaxsd	%ymm5, %ymm4, %ymm2
100006508: c5 f5 6b ca                 	vpackssdw	%ymm2, %ymm1, %ymm1
10000650c: c5 fd 6f b4 24 40 02 00 00  	vmovdqa	576(%rsp), %ymm6
100006515: c5 ed 73 d6 01              	vpsrlq	$1, %ymm6, %ymm2
10000651a: c5 fd 6f ac 24 a0 02 00 00  	vmovdqa	672(%rsp), %ymm5
100006523: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006527: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000652c: c5 fd 6f a4 24 80 02 00 00  	vmovdqa	640(%rsp), %ymm4
100006535: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006539: c4 c1 f9 7e d2              	vmovq	%xmm2, %r10
10000653e: c4 e3 f9 16 d0 01           	vpextrq	$1, %xmm2, %rax
100006544: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000654a: c4 c1 f9 7e d0              	vmovq	%xmm2, %r8
10000654f: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
100006555: c5 fd 6f bc 24 20 02 00 00  	vmovdqa	544(%rsp), %ymm7
10000655e: c5 ed 73 d7 01              	vpsrlq	$1, %ymm7, %ymm2
100006563: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006567: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000656c: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006570: c4 c1 f9 7e d5              	vmovq	%xmm2, %r13
100006575: c4 c3 f9 16 d6 01           	vpextrq	$1, %xmm2, %r14
10000657b: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006581: c4 e1 f9 7e d6              	vmovq	%xmm2, %rsi
100006586: c4 e3 f9 16 d7 01           	vpextrq	$1, %xmm2, %rdi
10000658c: c5 7d 6f 84 24 00 02 00 00  	vmovdqa	512(%rsp), %ymm8
100006595: c4 c1 6d 73 d0 01           	vpsrlq	$1, %ymm8, %ymm2
10000659b: c4 e3 fd 00 c0 d8           	vpermq	$216, %ymm0, %ymm0
1000065a1: c4 e3 fd 00 c9 d8           	vpermq	$216, %ymm1, %ymm1
1000065a7: c5 f5 63 c0                 	vpacksswb	%ymm0, %ymm1, %ymm0
1000065ab: c5 7d 6f 8c 24 e0 01 00 00  	vmovdqa	480(%rsp), %ymm9
1000065b4: c4 c1 65 73 d1 01           	vpsrlq	$1, %ymm9, %ymm3
1000065ba: c5 ed d4 cd                 	vpaddq	%ymm5, %ymm2, %ymm1
1000065be: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
1000065c3: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000065c7: c4 e3 f9 16 8c 24 40 01 00 00 01    	vpextrq	$1, %xmm1, 320(%rsp)
1000065d2: c4 e1 f9 7e ca              	vmovq	%xmm1, %rdx
1000065d7: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
1000065dd: c5 f9 d6 8c 24 20 01 00 00  	vmovq	%xmm1, 288(%rsp)
1000065e6: c4 c3 f9 16 cc 01           	vpextrq	$1, %xmm1, %r12
1000065ec: c5 7d 6f 94 24 c0 01 00 00  	vmovdqa	448(%rsp), %ymm10
1000065f5: c4 c1 75 73 d2 01           	vpsrlq	$1, %ymm10, %ymm1
1000065fb: c5 e5 d4 d5                 	vpaddq	%ymm5, %ymm3, %ymm2
1000065ff: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006604: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006608: c4 83 79 14 04 17 00        	vpextrb	$0, %xmm0, (%r15,%r10)
10000660f: c4 e1 f9 7e d1              	vmovq	%xmm2, %rcx
100006614: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
10000661a: c4 c3 79 14 04 07 01        	vpextrb	$1, %xmm0, (%r15,%rax)
100006621: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006627: c4 e3 f9 16 94 24 00 01 00 00 01    	vpextrq	$1, %xmm2, 256(%rsp)
100006632: c4 83 79 14 04 07 02        	vpextrb	$2, %xmm0, (%r15,%r8)
100006639: c4 e1 f9 7e d3              	vmovq	%xmm2, %rbx
10000663e: c5 fd 6f 9c 24 a0 01 00 00  	vmovdqa	416(%rsp), %ymm3
100006647: c5 ed 73 d3 01              	vpsrlq	$1, %ymm3, %ymm2
10000664c: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006650: c5 f5 d4 cd                 	vpaddq	%ymm5, %ymm1, %ymm1
100006654: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
100006659: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000665e: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006662: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100006666: c4 83 79 14 04 0f 03        	vpextrb	$3, %xmm0, (%r15,%r9)
10000666d: c4 c1 f9 7e c8              	vmovq	%xmm1, %r8
100006672: c4 83 79 14 04 2f 04        	vpextrb	$4, %xmm0, (%r15,%r13)
100006679: c4 e3 f9 16 8c 24 c0 00 00 00 01    	vpextrq	$1, %xmm1, 192(%rsp)
100006684: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
10000668a: c4 83 79 14 04 37 05        	vpextrb	$5, %xmm0, (%r15,%r14)
100006691: c4 c3 79 14 04 37 06        	vpextrb	$6, %xmm0, (%r15,%rsi)
100006698: c4 c1 f9 7e ce              	vmovq	%xmm1, %r14
10000669d: c4 e3 f9 16 8c 24 e0 00 00 00 01    	vpextrq	$1, %xmm1, 224(%rsp)
1000066a8: c4 c3 79 14 04 3f 07        	vpextrb	$7, %xmm0, (%r15,%rdi)
1000066af: c4 e3 f9 16 94 24 a0 00 00 00 01    	vpextrq	$1, %xmm2, 160(%rsp)
1000066ba: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
1000066c0: c4 c3 79 14 0c 17 00        	vpextrb	$0, %xmm1, (%r15,%rdx)
1000066c7: c4 e1 f9 7e d7              	vmovq	%xmm2, %rdi
1000066cc: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000066d2: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
1000066da: c4 c3 79 14 0c 07 01        	vpextrb	$1, %xmm1, (%r15,%rax)
1000066e1: c4 c1 f9 7e d5              	vmovq	%xmm2, %r13
1000066e6: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
1000066ee: c4 c3 79 14 0c 07 02        	vpextrb	$2, %xmm1, (%r15,%rax)
1000066f5: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
1000066fb: c5 7d 6f 9c 24 80 01 00 00  	vmovdqa	384(%rsp), %ymm11
100006704: c4 c1 6d 73 d3 01           	vpsrlq	$1, %ymm11, %ymm2
10000670a: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
10000670e: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006713: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006717: c4 83 79 14 0c 27 03        	vpextrb	$3, %xmm1, (%r15,%r12)
10000671e: c4 e1 f9 7e d2              	vmovq	%xmm2, %rdx
100006723: c4 c3 79 14 0c 0f 04        	vpextrb	$4, %xmm1, (%r15,%rcx)
10000672a: c4 e3 f9 16 d1 01           	vpextrq	$1, %xmm2, %rcx
100006730: c4 83 79 14 0c 17 05        	vpextrb	$5, %xmm1, (%r15,%r10)
100006737: c4 c3 79 14 0c 1f 06        	vpextrb	$6, %xmm1, (%r15,%rbx)
10000673e: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006744: c4 e1 f9 7e d3              	vmovq	%xmm2, %rbx
100006749: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
10000674f: c5 7d 6f a4 24 60 01 00 00  	vmovdqa	352(%rsp), %ymm12
100006758: c4 c1 6d 73 d4 01           	vpsrlq	$1, %ymm12, %ymm2
10000675e: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006762: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006767: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
10000676b: 48 8b 84 24 00 01 00 00     	movq	256(%rsp), %rax
100006773: c4 c3 79 14 0c 07 07        	vpextrb	$7, %xmm1, (%r15,%rax)
10000677a: c4 e1 f9 7e d0              	vmovq	%xmm2, %rax
10000677f: c4 83 79 14 04 07 08        	vpextrb	$8, %xmm0, (%r15,%r8)
100006786: c4 c3 f9 16 d0 01           	vpextrq	$1, %xmm2, %r8
10000678c: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006792: 48 8b b4 24 c0 00 00 00     	movq	192(%rsp), %rsi
10000679a: c4 c3 79 14 04 37 09        	vpextrb	$9, %xmm0, (%r15,%rsi)
1000067a1: c4 83 79 14 04 37 0a        	vpextrb	$10, %xmm0, (%r15,%r14)
1000067a8: c4 c1 f9 7e d6              	vmovq	%xmm2, %r14
1000067ad: c4 c3 f9 16 d4 01           	vpextrq	$1, %xmm2, %r12
1000067b3: c5 fd 6f 15 25 0b 00 00     	vmovdqa	2853(%rip), %ymm2
1000067bb: 48 8b b4 24 e0 00 00 00     	movq	224(%rsp), %rsi
1000067c3: c4 c3 79 14 04 37 0b        	vpextrb	$11, %xmm0, (%r15,%rsi)
1000067ca: c4 c3 79 14 04 3f 0c        	vpextrb	$12, %xmm0, (%r15,%rdi)
1000067d1: 48 8b b4 24 a0 00 00 00     	movq	160(%rsp), %rsi
1000067d9: c4 c3 79 14 04 37 0d        	vpextrb	$13, %xmm0, (%r15,%rsi)
1000067e0: c4 83 79 14 04 2f 0e        	vpextrb	$14, %xmm0, (%r15,%r13)
1000067e7: c4 83 79 14 04 0f 0f        	vpextrb	$15, %xmm0, (%r15,%r9)
1000067ee: c4 c3 79 14 0c 17 08        	vpextrb	$8, %xmm1, (%r15,%rdx)
1000067f5: c4 c3 79 14 0c 0f 09        	vpextrb	$9, %xmm1, (%r15,%rcx)
1000067fc: c4 c3 79 14 0c 1f 0a        	vpextrb	$10, %xmm1, (%r15,%rbx)
100006803: c4 83 79 14 0c 17 0b        	vpextrb	$11, %xmm1, (%r15,%r10)
10000680a: c4 c3 79 14 0c 07 0c        	vpextrb	$12, %xmm1, (%r15,%rax)
100006811: c4 83 79 14 0c 07 0d        	vpextrb	$13, %xmm1, (%r15,%r8)
100006818: c4 83 79 14 0c 37 0e        	vpextrb	$14, %xmm1, (%r15,%r14)
10000681f: c4 83 79 14 0c 27 0f        	vpextrb	$15, %xmm1, (%r15,%r12)
100006826: c4 e2 7d 59 05 b9 24 00 00  	vpbroadcastq	9401(%rip), %ymm0
10000682f: c5 cd d4 f0                 	vpaddq	%ymm0, %ymm6, %ymm6
100006833: c5 fd 7f b4 24 40 02 00 00  	vmovdqa	%ymm6, 576(%rsp)
10000683c: c5 c5 d4 f8                 	vpaddq	%ymm0, %ymm7, %ymm7
100006840: c5 fd 7f bc 24 20 02 00 00  	vmovdqa	%ymm7, 544(%rsp)
100006849: c5 3d d4 c0                 	vpaddq	%ymm0, %ymm8, %ymm8
10000684d: c5 7d 7f 84 24 00 02 00 00  	vmovdqa	%ymm8, 512(%rsp)
100006856: c5 35 d4 c8                 	vpaddq	%ymm0, %ymm9, %ymm9
10000685a: c5 7d 7f 8c 24 e0 01 00 00  	vmovdqa	%ymm9, 480(%rsp)
100006863: c5 2d d4 d0                 	vpaddq	%ymm0, %ymm10, %ymm10
100006867: c5 7d 7f 94 24 c0 01 00 00  	vmovdqa	%ymm10, 448(%rsp)
100006870: c5 e5 d4 d8                 	vpaddq	%ymm0, %ymm3, %ymm3
100006874: c5 fd 7f 9c 24 a0 01 00 00  	vmovdqa	%ymm3, 416(%rsp)
10000687d: c5 25 d4 d8                 	vpaddq	%ymm0, %ymm11, %ymm11
100006881: c5 7d 7f 9c 24 80 01 00 00  	vmovdqa	%ymm11, 384(%rsp)
10000688a: c5 1d d4 e0                 	vpaddq	%ymm0, %ymm12, %ymm12
10000688e: c5 7d 7f a4 24 60 01 00 00  	vmovdqa	%ymm12, 352(%rsp)
100006897: 49 83 c3 20                 	addq	$32, %r11
10000689b: 49 81 fb e0 00 00 00        	cmpq	$224, %r11
1000068a2: 0f 85 68 f6 ff ff           	jne	-2456 <__ZN11LineNetwork7forwardEv+0x840>
1000068a8: ba c0 01 00 00              	movl	$448, %edx
1000068ad: 44 8b 44 24 14              	movl	20(%rsp), %r8d
1000068b2: 48 8b 74 24 58              	movq	88(%rsp), %rsi
1000068b7: 4c 8b 6c 24 68              	movq	104(%rsp), %r13
1000068bc: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
1000068c1: eb 0f                       	jmp	15 <__ZN11LineNetwork7forwardEv+0x1202>
1000068c3: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000068cd: 0f 1f 00                    	nopl	(%rax)
1000068d0: 31 d2                       	xorl	%edx, %edx
1000068d2: 48 83 44 24 08 02           	addq	$2, 8(%rsp)
1000068d8: 48 89 d0                    	movq	%rdx, %rax
1000068db: 48 d1 e8                    	shrq	%rax
1000068de: 4c 8b b4 24 98 00 00 00     	movq	152(%rsp), %r14
1000068e6: 4c 01 f0                    	addq	%r14, %rax
1000068e9: 4c 8d 0c c7                 	leaq	(%rdi,%rax,8), %r9
1000068ed: 4c 8b 54 24 20              	movq	32(%rsp), %r10
1000068f2: 4c 8b 5c 24 18              	movq	24(%rsp), %r11
1000068f7: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x1248>
1000068f9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100006900: 41 88 09                    	movb	%cl, (%r9)
100006903: 48 83 c2 02                 	addq	$2, %rdx
100006907: 49 83 c1 08                 	addq	$8, %r9
10000690b: 48 81 fa fd 01 00 00        	cmpq	$509, %rdx
100006912: 0f 83 78 f4 ff ff           	jae	-2952 <__ZN11LineNetwork7forwardEv+0x6c0>
100006918: 4c 8b 64 24 30              	movq	48(%rsp), %r12
10000691d: 41 0f be 8c 14 fe fb ff ff  	movsbl	-1026(%r12,%rdx), %ecx
100006926: 41 0f be 02                 	movsbl	(%r10), %eax
10000692a: 0f af c1                    	imull	%ecx, %eax
10000692d: 41 0f be 8c 14 ff fb ff ff  	movsbl	-1025(%r12,%rdx), %ecx
100006936: 41 0f be 5a 01              	movsbl	1(%r10), %ebx
10000693b: 0f af d9                    	imull	%ecx, %ebx
10000693e: 01 c3                       	addl	%eax, %ebx
100006940: 41 0f be 8c 14 00 fc ff ff  	movsbl	-1024(%r12,%rdx), %ecx
100006949: 41 0f be 42 02              	movsbl	2(%r10), %eax
10000694e: 0f af c1                    	imull	%ecx, %eax
100006951: 01 d8                       	addl	%ebx, %eax
100006953: 41 0f be 8c 14 fe fd ff ff  	movsbl	-514(%r12,%rdx), %ecx
10000695c: 41 0f be 5a 03              	movsbl	3(%r10), %ebx
100006961: 0f af d9                    	imull	%ecx, %ebx
100006964: 01 c3                       	addl	%eax, %ebx
100006966: 41 0f be 8c 14 ff fd ff ff  	movsbl	-513(%r12,%rdx), %ecx
10000696f: 41 0f be 42 04              	movsbl	4(%r10), %eax
100006974: 0f af c1                    	imull	%ecx, %eax
100006977: 01 d8                       	addl	%ebx, %eax
100006979: 41 0f be 8c 14 00 fe ff ff  	movsbl	-512(%r12,%rdx), %ecx
100006982: 41 0f be 5a 05              	movsbl	5(%r10), %ebx
100006987: 0f af d9                    	imull	%ecx, %ebx
10000698a: 01 c3                       	addl	%eax, %ebx
10000698c: 41 0f be 4c 14 fe           	movsbl	-2(%r12,%rdx), %ecx
100006992: 41 0f be 42 06              	movsbl	6(%r10), %eax
100006997: 0f af c1                    	imull	%ecx, %eax
10000699a: 01 d8                       	addl	%ebx, %eax
10000699c: 41 0f be 4c 14 ff           	movsbl	-1(%r12,%rdx), %ecx
1000069a2: 41 0f be 5a 07              	movsbl	7(%r10), %ebx
1000069a7: 0f af d9                    	imull	%ecx, %ebx
1000069aa: 01 c3                       	addl	%eax, %ebx
1000069ac: 41 0f be 0c 14              	movsbl	(%r12,%rdx), %ecx
1000069b1: 41 0f be 42 08              	movsbl	8(%r10), %eax
1000069b6: 0f af c1                    	imull	%ecx, %eax
1000069b9: 01 d8                       	addl	%ebx, %eax
1000069bb: 41 0f be 1b                 	movsbl	(%r11), %ebx
1000069bf: 01 c3                       	addl	%eax, %ebx
1000069c1: 41 0f af d8                 	imull	%r8d, %ebx
1000069c5: 89 d9                       	movl	%ebx, %ecx
1000069c7: c1 f9 1f                    	sarl	$31, %ecx
1000069ca: c1 e9 12                    	shrl	$18, %ecx
1000069cd: 01 d9                       	addl	%ebx, %ecx
1000069cf: c1 f9 0e                    	sarl	$14, %ecx
1000069d2: 81 f9 80 00 00 00           	cmpl	$128, %ecx
1000069d8: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x130f>
1000069da: b9 7f 00 00 00              	movl	$127, %ecx
1000069df: 83 f9 81                    	cmpl	$-127, %ecx
1000069e2: 0f 8f 18 ff ff ff           	jg	-232 <__ZN11LineNetwork7forwardEv+0x1230>
1000069e8: b9 81 00 00 00              	movl	$129, %ecx
1000069ed: e9 0e ff ff ff              	jmp	-242 <__ZN11LineNetwork7forwardEv+0x1230>
1000069f2: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
1000069f6: 5b                          	popq	%rbx
1000069f7: 41 5c                       	popq	%r12
1000069f9: 41 5d                       	popq	%r13
1000069fb: 41 5e                       	popq	%r14
1000069fd: 41 5f                       	popq	%r15
1000069ff: 5d                          	popq	%rbp
100006a00: c5 f8 77                    	vzeroupper
100006a03: c3                          	retq
100006a04: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100006a0e: 66 90                       	nop
100006a10: 55                          	pushq	%rbp
100006a11: 48 89 e5                    	movq	%rsp, %rbp
100006a14: 5d                          	popq	%rbp
100006a15: e9 16 df ff ff              	jmp	-8426 <__ZN14ModelInterfaceD2Ev>
100006a1a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100006a20: 55                          	pushq	%rbp
100006a21: 48 89 e5                    	movq	%rsp, %rbp
100006a24: 53                          	pushq	%rbx
100006a25: 50                          	pushq	%rax
100006a26: 48 89 fb                    	movq	%rdi, %rbx
100006a29: e8 02 df ff ff              	callq	-8446 <__ZN14ModelInterfaceD2Ev>
100006a2e: 48 89 df                    	movq	%rbx, %rdi
100006a31: 48 83 c4 08                 	addq	$8, %rsp
100006a35: 5b                          	popq	%rbx
100006a36: 5d                          	popq	%rbp
100006a37: e9 60 04 00 00              	jmp	1120 <dyld_stub_binder+0x100006e9c>
100006a3c: 0f 1f 40 00                 	nopl	(%rax)
100006a40: 55                          	pushq	%rbp
100006a41: 48 89 e5                    	movq	%rsp, %rbp
100006a44: 48 83 e4 e0                 	andq	$-32, %rsp
100006a48: 48 81 ec a0 00 00 00        	subq	$160, %rsp
100006a4f: 48 8b 05 0a 26 00 00        	movq	9738(%rip), %rax
100006a56: 48 8b 00                    	movq	(%rax), %rax
100006a59: 48 89 84 24 88 00 00 00     	movq	%rax, 136(%rsp)
100006a61: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100006a65: c5 fc 29 44 24 60           	vmovaps	%ymm0, 96(%rsp)
100006a6b: c5 fc 29 44 24 40           	vmovaps	%ymm0, 64(%rsp)
100006a71: c5 fc 29 44 24 20           	vmovaps	%ymm0, 32(%rsp)
100006a77: c5 fc 29 04 24              	vmovaps	%ymm0, (%rsp)
100006a7c: c4 e2 7d 21 17              	vpmovsxbd	(%rdi), %ymm2
100006a81: c4 e2 7d 21 1e              	vpmovsxbd	(%rsi), %ymm3
100006a86: c4 e2 7d 21 47 08           	vpmovsxbd	8(%rdi), %ymm0
100006a8c: c4 e2 7d 21 4e 08           	vpmovsxbd	8(%rsi), %ymm1
100006a92: c4 e2 75 40 c0              	vpmulld	%ymm0, %ymm1, %ymm0
100006a97: c4 e2 7d 21 67 10           	vpmovsxbd	16(%rdi), %ymm4
100006a9d: c4 e2 7d 21 6e 10           	vpmovsxbd	16(%rsi), %ymm5
100006aa3: c4 e2 7d 21 4f 18           	vpmovsxbd	24(%rdi), %ymm1
100006aa9: c4 e2 7d 21 76 18           	vpmovsxbd	24(%rsi), %ymm6
100006aaf: c4 e2 4d 40 c9              	vpmulld	%ymm1, %ymm6, %ymm1
100006ab4: c5 f5 fe c8                 	vpaddd	%ymm0, %ymm1, %ymm1
100006ab8: c5 fd 7f 4c 24 20           	vmovdqa	%ymm1, 32(%rsp)
100006abe: c4 e2 7d 21 77 20           	vpmovsxbd	32(%rdi), %ymm6
100006ac4: c4 e2 7d 21 7e 20           	vpmovsxbd	32(%rsi), %ymm7
100006aca: c5 fd 6f 44 24 40           	vmovdqa	64(%rsp), %ymm0
100006ad0: c5 fd fe 44 24 60           	vpaddd	96(%rsp), %ymm0, %ymm0
100006ad6: 48 8b 05 83 25 00 00        	movq	9603(%rip), %rax
100006add: 48 8b 00                    	movq	(%rax), %rax
100006ae0: 48 3b 84 24 88 00 00 00     	cmpq	136(%rsp), %rax
100006ae8: 0f 85 c3 00 00 00           	jne	195 <__ZN11LineNetwork7forwardEv+0x14e1>
100006aee: c4 e2 65 40 d2              	vpmulld	%ymm2, %ymm3, %ymm2
100006af3: c4 e2 55 40 dc              	vpmulld	%ymm4, %ymm5, %ymm3
100006af8: c5 e5 fe d2                 	vpaddd	%ymm2, %ymm3, %ymm2
100006afc: c4 e2 45 40 de              	vpmulld	%ymm6, %ymm7, %ymm3
100006b01: c5 e5 fe d2                 	vpaddd	%ymm2, %ymm3, %ymm2
100006b05: c5 f9 7e d0                 	vmovd	%xmm2, %eax
100006b09: c4 e3 79 16 d1 01           	vpextrd	$1, %xmm2, %ecx
100006b0f: c4 e3 79 16 d2 02           	vpextrd	$2, %xmm2, %edx
100006b15: 01 c1                       	addl	%eax, %ecx
100006b17: 01 ca                       	addl	%ecx, %edx
100006b19: c4 e3 79 16 d0 03           	vpextrd	$3, %xmm2, %eax
100006b1f: 01 d0                       	addl	%edx, %eax
100006b21: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006b27: c5 f9 7e d1                 	vmovd	%xmm2, %ecx
100006b2b: 01 c1                       	addl	%eax, %ecx
100006b2d: c4 e3 79 16 d0 01           	vpextrd	$1, %xmm2, %eax
100006b33: c4 e3 79 16 d2 02           	vpextrd	$2, %xmm2, %edx
100006b39: 01 c8                       	addl	%ecx, %eax
100006b3b: 01 c2                       	addl	%eax, %edx
100006b3d: c4 e3 79 16 d0 03           	vpextrd	$3, %xmm2, %eax
100006b43: 01 d0                       	addl	%edx, %eax
100006b45: c5 f9 7e c9                 	vmovd	%xmm1, %ecx
100006b49: 01 c1                       	addl	%eax, %ecx
100006b4b: c4 e3 79 16 c8 01           	vpextrd	$1, %xmm1, %eax
100006b51: 01 c8                       	addl	%ecx, %eax
100006b53: c4 e3 79 16 c9 02           	vpextrd	$2, %xmm1, %ecx
100006b59: 01 c1                       	addl	%eax, %ecx
100006b5b: c4 e3 79 16 c8 03           	vpextrd	$3, %xmm1, %eax
100006b61: 01 c8                       	addl	%ecx, %eax
100006b63: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
100006b69: c5 f9 7e c9                 	vmovd	%xmm1, %ecx
100006b6d: 01 c1                       	addl	%eax, %ecx
100006b6f: c4 e3 79 16 c8 01           	vpextrd	$1, %xmm1, %eax
100006b75: 01 c8                       	addl	%ecx, %eax
100006b77: c4 e3 79 16 c9 02           	vpextrd	$2, %xmm1, %ecx
100006b7d: 01 c1                       	addl	%eax, %ecx
100006b7f: c4 e3 79 16 ca 03           	vpextrd	$3, %xmm1, %edx
100006b85: 01 ca                       	addl	%ecx, %edx
100006b87: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006b8d: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006b91: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006b96: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006b9a: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006b9f: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006ba3: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006ba7: 01 d0                       	addl	%edx, %eax
100006ba9: 48 89 ec                    	movq	%rbp, %rsp
100006bac: 5d                          	popq	%rbp
100006bad: c5 f8 77                    	vzeroupper
100006bb0: c3                          	retq
100006bb1: c5 f8 77                    	vzeroupper
100006bb4: e8 01 03 00 00              	callq	769 <dyld_stub_binder+0x100006eba>
100006bb9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100006bc0: 55                          	pushq	%rbp
100006bc1: 48 89 e5                    	movq	%rsp, %rbp
100006bc4: 48 83 e4 e0                 	andq	$-32, %rsp
100006bc8: 48 81 ec a0 00 00 00        	subq	$160, %rsp
100006bcf: 48 8b 05 8a 24 00 00        	movq	9354(%rip), %rax
100006bd6: 48 8b 00                    	movq	(%rax), %rax
100006bd9: 48 89 84 24 88 00 00 00     	movq	%rax, 136(%rsp)
100006be1: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100006be5: c5 fc 29 44 24 60           	vmovaps	%ymm0, 96(%rsp)
100006beb: c5 fc 29 44 24 40           	vmovaps	%ymm0, 64(%rsp)
100006bf1: c5 fc 29 44 24 20           	vmovaps	%ymm0, 32(%rsp)
100006bf7: c5 fc 29 04 24              	vmovaps	%ymm0, (%rsp)
100006bfc: c4 e2 7d 21 07              	vpmovsxbd	(%rdi), %ymm0
100006c01: c4 e2 7d 21 0e              	vpmovsxbd	(%rsi), %ymm1
100006c06: c4 e2 75 40 d8              	vpmulld	%ymm0, %ymm1, %ymm3
100006c0b: c4 e2 7d 21 47 08           	vpmovsxbd	8(%rdi), %ymm0
100006c11: c4 e2 7d 21 4e 08           	vpmovsxbd	8(%rsi), %ymm1
100006c17: c4 e2 75 40 d0              	vpmulld	%ymm0, %ymm1, %ymm2
100006c1c: c4 e2 7d 21 47 10           	vpmovsxbd	16(%rdi), %ymm0
100006c22: c4 e2 7d 21 4e 10           	vpmovsxbd	16(%rsi), %ymm1
100006c28: c4 e2 75 40 c0              	vpmulld	%ymm0, %ymm1, %ymm0
100006c2d: c4 e2 7d 21 4f 18           	vpmovsxbd	24(%rdi), %ymm1
100006c33: c4 e2 7d 21 66 18           	vpmovsxbd	24(%rsi), %ymm4
100006c39: c4 e2 5d 40 c9              	vpmulld	%ymm1, %ymm4, %ymm1
100006c3e: c5 fd 7f 1c 24              	vmovdqa	%ymm3, (%rsp)
100006c43: c5 fd 7f 54 24 20           	vmovdqa	%ymm2, 32(%rsp)
100006c49: c5 fd 7f 44 24 40           	vmovdqa	%ymm0, 64(%rsp)
100006c4f: c5 fd 7f 4c 24 60           	vmovdqa	%ymm1, 96(%rsp)
100006c55: c4 e2 7d 21 67 20           	vpmovsxbd	32(%rdi), %ymm4
100006c5b: c4 e2 7d 21 6e 20           	vpmovsxbd	32(%rsi), %ymm5
100006c61: c4 e2 7d 21 77 28           	vpmovsxbd	40(%rdi), %ymm6
100006c67: c4 e2 7d 21 7e 28           	vpmovsxbd	40(%rsi), %ymm7
100006c6d: c4 e2 45 40 f6              	vpmulld	%ymm6, %ymm7, %ymm6
100006c72: c5 cd fe d2                 	vpaddd	%ymm2, %ymm6, %ymm2
100006c76: c5 fd 7f 54 24 20           	vmovdqa	%ymm2, 32(%rsp)
100006c7c: c4 e2 7d 21 77 30           	vpmovsxbd	48(%rdi), %ymm6
100006c82: c4 e2 7d 21 7e 30           	vpmovsxbd	48(%rsi), %ymm7
100006c88: 48 8b 05 d1 23 00 00        	movq	9169(%rip), %rax
100006c8f: 48 8b 00                    	movq	(%rax), %rax
100006c92: 48 3b 84 24 88 00 00 00     	cmpq	136(%rsp), %rax
100006c9a: 0f 85 c2 00 00 00           	jne	194 <__ZN11LineNetwork7forwardEv+0x1692>
100006ca0: c4 e2 55 40 e4              	vpmulld	%ymm4, %ymm5, %ymm4
100006ca5: c5 dd fe db                 	vpaddd	%ymm3, %ymm4, %ymm3
100006ca9: c4 e2 45 40 e6              	vpmulld	%ymm6, %ymm7, %ymm4
100006cae: c5 dd fe db                 	vpaddd	%ymm3, %ymm4, %ymm3
100006cb2: c5 f9 7e d8                 	vmovd	%xmm3, %eax
100006cb6: c4 e3 79 16 d9 01           	vpextrd	$1, %xmm3, %ecx
100006cbc: 01 c1                       	addl	%eax, %ecx
100006cbe: c4 e3 79 16 d8 02           	vpextrd	$2, %xmm3, %eax
100006cc4: 01 c8                       	addl	%ecx, %eax
100006cc6: c4 e3 79 16 d9 03           	vpextrd	$3, %xmm3, %ecx
100006ccc: 01 c1                       	addl	%eax, %ecx
100006cce: c4 e3 7d 39 db 01           	vextracti128	$1, %ymm3, %xmm3
100006cd4: c5 f9 7e d8                 	vmovd	%xmm3, %eax
100006cd8: 01 c8                       	addl	%ecx, %eax
100006cda: c4 e3 79 16 d9 01           	vpextrd	$1, %xmm3, %ecx
100006ce0: 01 c1                       	addl	%eax, %ecx
100006ce2: c4 e3 79 16 d8 02           	vpextrd	$2, %xmm3, %eax
100006ce8: 01 c8                       	addl	%ecx, %eax
100006cea: c4 e3 79 16 d9 03           	vpextrd	$3, %xmm3, %ecx
100006cf0: 01 c1                       	addl	%eax, %ecx
100006cf2: c5 f9 7e d0                 	vmovd	%xmm2, %eax
100006cf6: 01 c8                       	addl	%ecx, %eax
100006cf8: c4 e3 79 16 d1 01           	vpextrd	$1, %xmm2, %ecx
100006cfe: 01 c1                       	addl	%eax, %ecx
100006d00: c4 e3 79 16 d0 02           	vpextrd	$2, %xmm2, %eax
100006d06: 01 c8                       	addl	%ecx, %eax
100006d08: c4 e3 79 16 d1 03           	vpextrd	$3, %xmm2, %ecx
100006d0e: 01 c1                       	addl	%eax, %ecx
100006d10: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006d16: c5 f9 7e d0                 	vmovd	%xmm2, %eax
100006d1a: 01 c8                       	addl	%ecx, %eax
100006d1c: c4 e3 79 16 d1 01           	vpextrd	$1, %xmm2, %ecx
100006d22: 01 c1                       	addl	%eax, %ecx
100006d24: c4 e3 79 16 d0 02           	vpextrd	$2, %xmm2, %eax
100006d2a: 01 c8                       	addl	%ecx, %eax
100006d2c: c4 e3 79 16 d1 03           	vpextrd	$3, %xmm2, %ecx
100006d32: 01 c1                       	addl	%eax, %ecx
100006d34: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006d38: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006d3e: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006d42: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006d47: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006d4b: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006d50: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006d54: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006d58: 01 c8                       	addl	%ecx, %eax
100006d5a: 48 89 ec                    	movq	%rbp, %rsp
100006d5d: 5d                          	popq	%rbp
100006d5e: c5 f8 77                    	vzeroupper
100006d61: c3                          	retq
100006d62: c5 f8 77                    	vzeroupper
100006d65: e8 50 01 00 00              	callq	336 <dyld_stub_binder+0x100006eba>
100006d6a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100006d70: 55                          	pushq	%rbp
100006d71: 48 89 e5                    	movq	%rsp, %rbp
100006d74: c4 e2 7d 21 47 08           	vpmovsxbd	8(%rdi), %ymm0
100006d7a: c4 e2 7d 21 4f 18           	vpmovsxbd	24(%rdi), %ymm1
100006d80: c4 e2 7d 21 17              	vpmovsxbd	(%rdi), %ymm2
100006d85: c4 e2 7d 21 5f 10           	vpmovsxbd	16(%rdi), %ymm3
100006d8b: c4 e2 7d 21 66 08           	vpmovsxbd	8(%rsi), %ymm4
100006d91: c4 e2 5d 40 c0              	vpmulld	%ymm0, %ymm4, %ymm0
100006d96: c4 e2 7d 21 66 18           	vpmovsxbd	24(%rsi), %ymm4
100006d9c: c4 e2 5d 40 c9              	vpmulld	%ymm1, %ymm4, %ymm1
100006da1: c4 e2 7d 21 26              	vpmovsxbd	(%rsi), %ymm4
100006da6: c4 e2 5d 40 d2              	vpmulld	%ymm2, %ymm4, %ymm2
100006dab: c4 e2 7d 21 66 10           	vpmovsxbd	16(%rsi), %ymm4
100006db1: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006db5: c4 e2 5d 40 cb              	vpmulld	%ymm3, %ymm4, %ymm1
100006dba: c5 ed fe c9                 	vpaddd	%ymm1, %ymm2, %ymm1
100006dbe: c5 f5 fe c0                 	vpaddd	%ymm0, %ymm1, %ymm0
100006dc2: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006dc8: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006dcc: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006dd1: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006dd5: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006dda: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006dde: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006de2: 5d                          	popq	%rbp
100006de3: c5 f8 77                    	vzeroupper
100006de6: c3                          	retq

Disassembly of section __TEXT,__stubs:

0000000100006de8 __stubs:
100006de8: ff 25 12 32 00 00           	jmpq	*12818(%rip)
100006dee: ff 25 14 32 00 00           	jmpq	*12820(%rip)
100006df4: ff 25 16 32 00 00           	jmpq	*12822(%rip)
100006dfa: ff 25 18 32 00 00           	jmpq	*12824(%rip)
100006e00: ff 25 1a 32 00 00           	jmpq	*12826(%rip)
100006e06: ff 25 1c 32 00 00           	jmpq	*12828(%rip)
100006e0c: ff 25 1e 32 00 00           	jmpq	*12830(%rip)
100006e12: ff 25 20 32 00 00           	jmpq	*12832(%rip)
100006e18: ff 25 22 32 00 00           	jmpq	*12834(%rip)
100006e1e: ff 25 24 32 00 00           	jmpq	*12836(%rip)
100006e24: ff 25 26 32 00 00           	jmpq	*12838(%rip)
100006e2a: ff 25 28 32 00 00           	jmpq	*12840(%rip)
100006e30: ff 25 2a 32 00 00           	jmpq	*12842(%rip)
100006e36: ff 25 2c 32 00 00           	jmpq	*12844(%rip)
100006e3c: ff 25 2e 32 00 00           	jmpq	*12846(%rip)
100006e42: ff 25 30 32 00 00           	jmpq	*12848(%rip)
100006e48: ff 25 32 32 00 00           	jmpq	*12850(%rip)
100006e4e: ff 25 34 32 00 00           	jmpq	*12852(%rip)
100006e54: ff 25 36 32 00 00           	jmpq	*12854(%rip)
100006e5a: ff 25 38 32 00 00           	jmpq	*12856(%rip)
100006e60: ff 25 3a 32 00 00           	jmpq	*12858(%rip)
100006e66: ff 25 3c 32 00 00           	jmpq	*12860(%rip)
100006e6c: ff 25 3e 32 00 00           	jmpq	*12862(%rip)
100006e72: ff 25 40 32 00 00           	jmpq	*12864(%rip)
100006e78: ff 25 42 32 00 00           	jmpq	*12866(%rip)
100006e7e: ff 25 44 32 00 00           	jmpq	*12868(%rip)
100006e84: ff 25 46 32 00 00           	jmpq	*12870(%rip)
100006e8a: ff 25 48 32 00 00           	jmpq	*12872(%rip)
100006e90: ff 25 4a 32 00 00           	jmpq	*12874(%rip)
100006e96: ff 25 4c 32 00 00           	jmpq	*12876(%rip)
100006e9c: ff 25 4e 32 00 00           	jmpq	*12878(%rip)
100006ea2: ff 25 50 32 00 00           	jmpq	*12880(%rip)
100006ea8: ff 25 52 32 00 00           	jmpq	*12882(%rip)
100006eae: ff 25 54 32 00 00           	jmpq	*12884(%rip)
100006eb4: ff 25 56 32 00 00           	jmpq	*12886(%rip)
100006eba: ff 25 58 32 00 00           	jmpq	*12888(%rip)
100006ec0: ff 25 5a 32 00 00           	jmpq	*12890(%rip)
100006ec6: ff 25 5c 32 00 00           	jmpq	*12892(%rip)

Disassembly of section __TEXT,__stub_helper:

0000000100006ecc __stub_helper:
100006ecc: 4c 8d 1d 5d 32 00 00        	leaq	12893(%rip), %r11
100006ed3: 41 53                       	pushq	%r11
100006ed5: ff 25 8d 21 00 00           	jmpq	*8589(%rip)
100006edb: 90                          	nop
100006edc: 68 4e 01 00 00              	pushq	$334
100006ee1: e9 e6 ff ff ff              	jmp	-26 <__stub_helper>
100006ee6: 68 9c 02 00 00              	pushq	$668
100006eeb: e9 dc ff ff ff              	jmp	-36 <__stub_helper>
100006ef0: 68 17 00 00 00              	pushq	$23
100006ef5: e9 d2 ff ff ff              	jmp	-46 <__stub_helper>
100006efa: 68 7a 00 00 00              	pushq	$122
100006eff: e9 c8 ff ff ff              	jmp	-56 <__stub_helper>
100006f04: 68 9b 00 00 00              	pushq	$155
100006f09: e9 be ff ff ff              	jmp	-66 <__stub_helper>
100006f0e: 68 2e 03 00 00              	pushq	$814
100006f13: e9 b4 ff ff ff              	jmp	-76 <__stub_helper>
100006f18: 68 b9 01 00 00              	pushq	$441
100006f1d: e9 aa ff ff ff              	jmp	-86 <__stub_helper>
100006f22: 68 07 02 00 00              	pushq	$519
100006f27: e9 a0 ff ff ff              	jmp	-96 <__stub_helper>
100006f2c: 68 b4 02 00 00              	pushq	$692
100006f31: e9 96 ff ff ff              	jmp	-106 <__stub_helper>
100006f36: 68 c4 00 00 00              	pushq	$196
100006f3b: e9 8c ff ff ff              	jmp	-116 <__stub_helper>
100006f40: 68 e5 00 00 00              	pushq	$229
100006f45: e9 82 ff ff ff              	jmp	-126 <__stub_helper>
100006f4a: 68 05 01 00 00              	pushq	$261
100006f4f: e9 78 ff ff ff              	jmp	-136 <__stub_helper>
100006f54: 68 27 01 00 00              	pushq	$295
100006f59: e9 6e ff ff ff              	jmp	-146 <__stub_helper>
100006f5e: 68 f6 02 00 00              	pushq	$758
100006f63: e9 64 ff ff ff              	jmp	-156 <__stub_helper>
100006f68: 68 11 03 00 00              	pushq	$785
100006f6d: e9 5a ff ff ff              	jmp	-166 <__stub_helper>
100006f72: 68 57 03 00 00              	pushq	$855
100006f77: e9 50 ff ff ff              	jmp	-176 <__stub_helper>
100006f7c: 68 86 03 00 00              	pushq	$902
100006f81: e9 46 ff ff ff              	jmp	-186 <__stub_helper>
100006f86: 68 ac 03 00 00              	pushq	$940
100006f8b: e9 3c ff ff ff              	jmp	-196 <__stub_helper>
100006f90: 68 00 04 00 00              	pushq	$1024
100006f95: e9 32 ff ff ff              	jmp	-206 <__stub_helper>
100006f9a: 68 55 04 00 00              	pushq	$1109
100006f9f: e9 28 ff ff ff              	jmp	-216 <__stub_helper>
100006fa4: 68 aa 04 00 00              	pushq	$1194
100006fa9: e9 1e ff ff ff              	jmp	-226 <__stub_helper>
100006fae: 68 f1 04 00 00              	pushq	$1265
100006fb3: e9 14 ff ff ff              	jmp	-236 <__stub_helper>
100006fb8: 68 35 05 00 00              	pushq	$1333
100006fbd: e9 0a ff ff ff              	jmp	-246 <__stub_helper>
100006fc2: 68 63 05 00 00              	pushq	$1379
100006fc7: e9 00 ff ff ff              	jmp	-256 <__stub_helper>
100006fcc: 68 81 05 00 00              	pushq	$1409
100006fd1: e9 f6 fe ff ff              	jmp	-266 <__stub_helper>
100006fd6: 68 c2 05 00 00              	pushq	$1474
100006fdb: e9 ec fe ff ff              	jmp	-276 <__stub_helper>
100006fe0: 68 e6 05 00 00              	pushq	$1510
100006fe5: e9 e2 fe ff ff              	jmp	-286 <__stub_helper>
100006fea: 68 05 06 00 00              	pushq	$1541
100006fef: e9 d8 fe ff ff              	jmp	-296 <__stub_helper>
100006ff4: 68 24 06 00 00              	pushq	$1572
100006ff9: e9 ce fe ff ff              	jmp	-306 <__stub_helper>
100006ffe: 68 3d 06 00 00              	pushq	$1597
100007003: e9 c4 fe ff ff              	jmp	-316 <__stub_helper>
100007008: 68 58 06 00 00              	pushq	$1624
10000700d: e9 ba fe ff ff              	jmp	-326 <__stub_helper>
100007012: 68 00 00 00 00              	pushq	$0
100007017: e9 b0 fe ff ff              	jmp	-336 <__stub_helper>
10000701c: 68 71 06 00 00              	pushq	$1649
100007021: e9 a6 fe ff ff              	jmp	-346 <__stub_helper>
100007026: 68 8b 06 00 00              	pushq	$1675
10000702b: e9 9c fe ff ff              	jmp	-356 <__stub_helper>
100007030: 68 9b 06 00 00              	pushq	$1691
100007035: e9 92 fe ff ff              	jmp	-366 <__stub_helper>
