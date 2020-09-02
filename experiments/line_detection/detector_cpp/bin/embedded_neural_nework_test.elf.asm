
bin/embedded_neural_nework_test.elf:	file format Mach-O 64-bit x86-64


Disassembly of section __TEXT,__text:

0000000100002640 __Z8get_timev:
100002640: 55                          	pushq	%rbp
100002641: 48 89 e5                    	movq	%rsp, %rbp
100002644: e8 1f 48 00 00              	callq	18463 <dyld_stub_binder+0x100006e68>
100002649: c4 e1 fb 2a c0              	vcvtsi2sd	%rax, %xmm0, %xmm0
10000264e: c5 fb 5e 05 ea 49 00 00     	vdivsd	18922(%rip), %xmm0, %xmm0
100002656: 5d                          	popq	%rbp
100002657: c3                          	retq
100002658: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100002660 __Z14get_predictionRN2cv3MatER14ModelInterfacef:
100002660: 55                          	pushq	%rbp
100002661: 48 89 e5                    	movq	%rsp, %rbp
100002664: 41 57                       	pushq	%r15
100002666: 41 56                       	pushq	%r14
100002668: 41 55                       	pushq	%r13
10000266a: 41 54                       	pushq	%r12
10000266c: 53                          	pushq	%rbx
10000266d: 48 81 ec 28 01 00 00        	subq	$296, %rsp
100002674: c5 fa 11 45 a8              	vmovss	%xmm0, -88(%rbp)
100002679: 49 89 d6                    	movq	%rdx, %r14
10000267c: 49 89 f4                    	movq	%rsi, %r12
10000267f: 48 89 fb                    	movq	%rdi, %rbx
100002682: 48 8b 05 cf 69 00 00        	movq	27087(%rip), %rax
100002689: 48 8b 00                    	movq	(%rax), %rax
10000268c: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100002690: 8b 46 08                    	movl	8(%rsi), %eax
100002693: 8b 4e 0c                    	movl	12(%rsi), %ecx
100002696: c7 85 d0 fe ff ff 00 00 ff 42       	movl	$1124007936, -304(%rbp)
1000026a0: 48 8d 95 d8 fe ff ff        	leaq	-296(%rbp), %rdx
1000026a7: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000026ab: c5 fc 11 85 d4 fe ff ff     	vmovups	%ymm0, -300(%rbp)
1000026b3: c5 fc 11 85 f0 fe ff ff     	vmovups	%ymm0, -272(%rbp)
1000026bb: 48 89 95 10 ff ff ff        	movq	%rdx, -240(%rbp)
1000026c2: 48 8d 95 20 ff ff ff        	leaq	-224(%rbp), %rdx
1000026c9: 48 89 95 18 ff ff ff        	movq	%rdx, -232(%rbp)
1000026d0: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000026d4: c5 f8 11 85 20 ff ff ff     	vmovups	%xmm0, -224(%rbp)
1000026dc: 89 4d b8                    	movl	%ecx, -72(%rbp)
1000026df: 89 45 bc                    	movl	%eax, -68(%rbp)
1000026e2: 4c 8d bd d0 fe ff ff        	leaq	-304(%rbp), %r15
1000026e9: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
1000026ed: 4c 89 ff                    	movq	%r15, %rdi
1000026f0: be 02 00 00 00              	movl	$2, %esi
1000026f5: 31 c9                       	xorl	%ecx, %ecx
1000026f7: c5 f8 77                    	vzeroupper
1000026fa: e8 fd 46 00 00              	callq	18173 <dyld_stub_binder+0x100006dfc>
1000026ff: 48 c7 85 40 ff ff ff 00 00 00 00    	movq	$0, -192(%rbp)
10000270a: c7 85 30 ff ff ff 00 00 01 01       	movl	$16842752, -208(%rbp)
100002714: 4c 89 a5 38 ff ff ff        	movq	%r12, -200(%rbp)
10000271b: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
100002723: c7 45 b8 00 00 01 02        	movl	$33619968, -72(%rbp)
10000272a: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
10000272e: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002735: 48 8d 75 b8                 	leaq	-72(%rbp), %rsi
100002739: ba 06 00 00 00              	movl	$6, %edx
10000273e: 31 c9                       	xorl	%ecx, %ecx
100002740: e8 e1 46 00 00              	callq	18145 <dyld_stub_binder+0x100006e26>
100002745: 41 8b 44 24 08              	movl	8(%r12), %eax
10000274a: 41 8b 4c 24 0c              	movl	12(%r12), %ecx
10000274f: c7 85 30 ff ff ff 00 00 ff 42       	movl	$1124007936, -208(%rbp)
100002759: 48 8d 95 38 ff ff ff        	leaq	-200(%rbp), %rdx
100002760: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002764: c5 fc 11 85 34 ff ff ff     	vmovups	%ymm0, -204(%rbp)
10000276c: c5 fc 11 85 50 ff ff ff     	vmovups	%ymm0, -176(%rbp)
100002774: 48 89 95 70 ff ff ff        	movq	%rdx, -144(%rbp)
10000277b: 48 8d 55 80                 	leaq	-128(%rbp), %rdx
10000277f: 48 89 95 78 ff ff ff        	movq	%rdx, -136(%rbp)
100002786: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000278a: c5 f8 11 45 80              	vmovups	%xmm0, -128(%rbp)
10000278f: 89 4d b8                    	movl	%ecx, -72(%rbp)
100002792: 89 45 bc                    	movl	%eax, -68(%rbp)
100002795: 4c 8d a5 30 ff ff ff        	leaq	-208(%rbp), %r12
10000279c: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
1000027a0: 4c 89 e7                    	movq	%r12, %rdi
1000027a3: be 02 00 00 00              	movl	$2, %esi
1000027a8: 31 c9                       	xorl	%ecx, %ecx
1000027aa: c5 f8 77                    	vzeroupper
1000027ad: e8 4a 46 00 00              	callq	17994 <dyld_stub_binder+0x100006dfc>
1000027b2: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
1000027ba: c7 45 b8 00 00 01 01        	movl	$16842752, -72(%rbp)
1000027c1: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
1000027c5: 48 c7 85 c0 fe ff ff 00 00 00 00    	movq	$0, -320(%rbp)
1000027d0: c7 85 b0 fe ff ff 00 00 01 02       	movl	$33619968, -336(%rbp)
1000027da: 4c 89 a5 b8 fe ff ff        	movq	%r12, -328(%rbp)
1000027e1: 41 8b 46 0c                 	movl	12(%r14), %eax
1000027e5: 41 8b 4e 10                 	movl	16(%r14), %ecx
1000027e9: 89 4d 90                    	movl	%ecx, -112(%rbp)
1000027ec: 89 45 94                    	movl	%eax, -108(%rbp)
1000027ef: 48 8d 7d b8                 	leaq	-72(%rbp), %rdi
1000027f3: 48 8d b5 b0 fe ff ff        	leaq	-336(%rbp), %rsi
1000027fa: 48 8d 55 90                 	leaq	-112(%rbp), %rdx
1000027fe: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002802: c5 f0 57 c9                 	vxorps	%xmm1, %xmm1, %xmm1
100002806: b9 03 00 00 00              	movl	$3, %ecx
10000280b: e8 04 46 00 00              	callq	17924 <dyld_stub_binder+0x100006e14>
100002810: 41 8b 46 0c                 	movl	12(%r14), %eax
100002814: 85 c0                       	testl	%eax, %eax
100002816: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
10000281a: 4d 89 f7                    	movq	%r14, %r15
10000281d: 0f 84 7c 00 00 00           	je	124 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x23f>
100002823: 41 8b 4f 10                 	movl	16(%r15), %ecx
100002827: 31 d2                       	xorl	%edx, %edx
100002829: 45 31 e4                    	xorl	%r12d, %r12d
10000282c: 85 c9                       	testl	%ecx, %ecx
10000282e: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1dc>
100002830: 31 c9                       	xorl	%ecx, %ecx
100002832: ff c2                       	incl	%edx
100002834: 39 c2                       	cmpl	%eax, %edx
100002836: 73 67                       	jae	103 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x23f>
100002838: 85 c9                       	testl	%ecx, %ecx
10000283a: 74 f4                       	je	-12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1d0>
10000283c: 89 55 a0                    	movl	%edx, -96(%rbp)
10000283f: 4c 63 f2                    	movslq	%edx, %r14
100002842: 45 31 ed                    	xorl	%r13d, %r13d
100002845: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000284f: 90                          	nop
100002850: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
100002857: 48 8b 00                    	movq	(%rax), %rax
10000285a: 49 0f af c6                 	imulq	%r14, %rax
10000285e: 48 03 85 40 ff ff ff        	addq	-192(%rbp), %rax
100002865: 49 63 cd                    	movslq	%r13d, %rcx
100002868: 0f b6 1c 01                 	movzbl	(%rcx,%rax), %ebx
10000286c: 4c 89 ff                    	movq	%r15, %rdi
10000286f: e8 6c 21 00 00              	callq	8556 <__ZN14ModelInterface12input_bufferEv>
100002874: 43 8d 0c 2c                 	leal	(%r12,%r13), %ecx
100002878: d0 eb                       	shrb	%bl
10000287a: 89 c9                       	movl	%ecx, %ecx
10000287c: 88 1c 08                    	movb	%bl, (%rax,%rcx)
10000287f: 41 ff c5                    	incl	%r13d
100002882: 41 8b 4f 10                 	movl	16(%r15), %ecx
100002886: 41 39 cd                    	cmpl	%ecx, %r13d
100002889: 72 c5                       	jb	-59 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1f0>
10000288b: 41 8b 47 0c                 	movl	12(%r15), %eax
10000288f: 45 01 ec                    	addl	%r13d, %r12d
100002892: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002896: 8b 55 a0                    	movl	-96(%rbp), %edx
100002899: ff c2                       	incl	%edx
10000289b: 39 c2                       	cmpl	%eax, %edx
10000289d: 72 99                       	jb	-103 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1d8>
10000289f: 49 8b 07                    	movq	(%r15), %rax
1000028a2: 4c 89 ff                    	movq	%r15, %rdi
1000028a5: ff 50 10                    	callq	*16(%rax)
1000028a8: 41 8b 47 18                 	movl	24(%r15), %eax
1000028ac: 41 8b 4f 1c                 	movl	28(%r15), %ecx
1000028b0: c7 03 00 00 ff 42           	movl	$1124007936, (%rbx)
1000028b6: 48 8d 53 08                 	leaq	8(%rbx), %rdx
1000028ba: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000028be: c5 fc 11 43 04              	vmovups	%ymm0, 4(%rbx)
1000028c3: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
1000028c8: 48 89 53 40                 	movq	%rdx, 64(%rbx)
1000028cc: 48 8d 53 50                 	leaq	80(%rbx), %rdx
1000028d0: 48 89 95 c8 fe ff ff        	movq	%rdx, -312(%rbp)
1000028d7: 48 89 53 48                 	movq	%rdx, 72(%rbx)
1000028db: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000028df: c5 f8 11 43 50              	vmovups	%xmm0, 80(%rbx)
1000028e4: 89 4d b8                    	movl	%ecx, -72(%rbp)
1000028e7: 89 45 bc                    	movl	%eax, -68(%rbp)
1000028ea: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
1000028ee: 48 89 df                    	movq	%rbx, %rdi
1000028f1: be 02 00 00 00              	movl	$2, %esi
1000028f6: 31 c9                       	xorl	%ecx, %ecx
1000028f8: c5 f8 77                    	vzeroupper
1000028fb: e8 fc 44 00 00              	callq	17660 <dyld_stub_binder+0x100006dfc>
100002900: 41 8b 47 18                 	movl	24(%r15), %eax
100002904: 41 83 7f 14 01              	cmpl	$1, 20(%r15)
100002909: 4d 89 fc                    	movq	%r15, %r12
10000290c: 0f 85 c7 00 00 00           	jne	199 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x379>
100002912: 85 c0                       	testl	%eax, %eax
100002914: 0f 84 e2 01 00 00           	je	482 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
10000291a: c5 fa 10 45 a8              	vmovss	-88(%rbp), %xmm0
10000291f: c5 fa 59 05 61 47 00 00     	vmulss	18273(%rip), %xmm0, %xmm0
100002927: c5 fa 11 45 a0              	vmovss	%xmm0, -96(%rbp)
10000292c: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002931: 45 31 ff                    	xorl	%r15d, %r15d
100002934: 31 d2                       	xorl	%edx, %edx
100002936: 45 31 ed                    	xorl	%r13d, %r13d
100002939: 85 c9                       	testl	%ecx, %ecx
10000293b: 75 13                       	jne	19 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2f0>
10000293d: 0f 1f 00                    	nopl	(%rax)
100002940: 31 c9                       	xorl	%ecx, %ecx
100002942: ff c2                       	incl	%edx
100002944: 39 c2                       	cmpl	%eax, %edx
100002946: 0f 83 b0 01 00 00           	jae	432 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
10000294c: 85 c9                       	testl	%ecx, %ecx
10000294e: 74 f0                       	je	-16 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2e0>
100002950: 89 55 a8                    	movl	%edx, -88(%rbp)
100002953: 4c 63 f2                    	movslq	%edx, %r14
100002956: 31 db                       	xorl	%ebx, %ebx
100002958: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002960: 4c 89 e7                    	movq	%r12, %rdi
100002963: e8 88 20 00 00              	callq	8328 <__ZN14ModelInterface13output_bufferEv>
100002968: 42 8d 0c 2b                 	leal	(%rbx,%r13), %ecx
10000296c: 89 c9                       	movl	%ecx, %ecx
10000296e: 0f b6 04 08                 	movzbl	(%rax,%rcx), %eax
100002972: 84 c0                       	testb	%al, %al
100002974: 41 0f 48 c7                 	cmovsl	%r15d, %eax
100002978: 0f be c8                    	movsbl	%al, %ecx
10000297b: c5 ea 2a c1                 	vcvtsi2ss	%ecx, %xmm2, %xmm0
10000297f: 48 8b 55 b0                 	movq	-80(%rbp), %rdx
100002983: 48 8b 4a 48                 	movq	72(%rdx), %rcx
100002987: 48 8b 09                    	movq	(%rcx), %rcx
10000298a: 49 0f af ce                 	imulq	%r14, %rcx
10000298e: 48 03 4a 10                 	addq	16(%rdx), %rcx
100002992: 48 63 db                    	movslq	%ebx, %rbx
100002995: 88 04 0b                    	movb	%al, (%rbx,%rcx)
100002998: 48 8b 42 48                 	movq	72(%rdx), %rax
10000299c: 48 8b 00                    	movq	(%rax), %rax
10000299f: 49 0f af c6                 	imulq	%r14, %rax
1000029a3: 48 03 42 10                 	addq	16(%rdx), %rax
1000029a7: c5 f8 2e 45 a0              	vucomiss	-96(%rbp), %xmm0
1000029ac: 0f 97 04 03                 	seta	(%rbx,%rax)
1000029b0: ff c3                       	incl	%ebx
1000029b2: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
1000029b7: 39 cb                       	cmpl	%ecx, %ebx
1000029b9: 72 a5                       	jb	-91 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x300>
1000029bb: 41 8b 44 24 18              	movl	24(%r12), %eax
1000029c0: 41 01 dd                    	addl	%ebx, %r13d
1000029c3: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
1000029c7: 8b 55 a8                    	movl	-88(%rbp), %edx
1000029ca: ff c2                       	incl	%edx
1000029cc: 39 c2                       	cmpl	%eax, %edx
1000029ce: 0f 82 78 ff ff ff           	jb	-136 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2ec>
1000029d4: e9 23 01 00 00              	jmp	291 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
1000029d9: 85 c0                       	testl	%eax, %eax
1000029db: 0f 84 1b 01 00 00           	je	283 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
1000029e1: c5 fa 10 45 a8              	vmovss	-88(%rbp), %xmm0
1000029e6: c5 fa 59 05 9a 46 00 00     	vmulss	18074(%rip), %xmm0, %xmm0
1000029ee: c5 fa 11 45 98              	vmovss	%xmm0, -104(%rbp)
1000029f3: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
1000029f8: 31 d2                       	xorl	%edx, %edx
1000029fa: 45 31 ff                    	xorl	%r15d, %r15d
1000029fd: 85 c9                       	testl	%ecx, %ecx
1000029ff: 75 29                       	jne	41 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3ca>
100002a01: e9 ea 00 00 00              	jmp	234 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x490>
100002a06: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002a10: 41 8b 44 24 18              	movl	24(%r12), %eax
100002a15: 8b 55 9c                    	movl	-100(%rbp), %edx
100002a18: ff c2                       	incl	%edx
100002a1a: 39 c2                       	cmpl	%eax, %edx
100002a1c: 0f 83 da 00 00 00           	jae	218 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002a22: 85 c9                       	testl	%ecx, %ecx
100002a24: 0f 84 c6 00 00 00           	je	198 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x490>
100002a2a: 89 55 9c                    	movl	%edx, -100(%rbp)
100002a2d: 48 63 c2                    	movslq	%edx, %rax
100002a30: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100002a34: 31 d2                       	xorl	%edx, %edx
100002a36: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100002a3a: 41 83 7c 24 14 00           	cmpl	$0, 20(%r12)
100002a40: 75 60                       	jne	96 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x442>
100002a42: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002a4c: 0f 1f 40 00                 	nopl	(%rax)
100002a50: 41 b6 81                    	movb	$-127, %r14b
100002a53: 45 31 ed                    	xorl	%r13d, %r13d
100002a56: 41 0f be c6                 	movsbl	%r14b, %eax
100002a5a: c5 ea 2a c0                 	vcvtsi2ss	%eax, %xmm2, %xmm0
100002a5e: c5 f8 2e 45 98              	vucomiss	-104(%rbp), %xmm0
100002a63: b8 00 00 00 00              	movl	$0, %eax
100002a68: 44 0f 46 e8                 	cmovbel	%eax, %r13d
100002a6c: 48 8b 43 48                 	movq	72(%rbx), %rax
100002a70: 48 8b 00                    	movq	(%rax), %rax
100002a73: 48 0f af 45 a8              	imulq	-88(%rbp), %rax
100002a78: 48 03 43 10                 	addq	16(%rbx), %rax
100002a7c: 48 8b 55 a0                 	movq	-96(%rbp), %rdx
100002a80: 48 63 d2                    	movslq	%edx, %rdx
100002a83: 44 88 2c 02                 	movb	%r13b, (%rdx,%rax)
100002a87: ff c2                       	incl	%edx
100002a89: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002a8e: 39 ca                       	cmpl	%ecx, %edx
100002a90: 0f 83 7a ff ff ff           	jae	-134 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3b0>
100002a96: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100002a9a: 41 83 7c 24 14 00           	cmpl	$0, 20(%r12)
100002aa0: 74 ae                       	je	-82 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3f0>
100002aa2: 41 b6 81                    	movb	$-127, %r14b
100002aa5: 31 db                       	xorl	%ebx, %ebx
100002aa7: 45 31 ed                    	xorl	%r13d, %r13d
100002aaa: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100002ab0: 4c 89 e7                    	movq	%r12, %rdi
100002ab3: e8 38 1f 00 00              	callq	7992 <__ZN14ModelInterface13output_bufferEv>
100002ab8: 41 8d 0c 1f                 	leal	(%r15,%rbx), %ecx
100002abc: 89 c9                       	movl	%ecx, %ecx
100002abe: 0f b6 04 08                 	movzbl	(%rax,%rcx), %eax
100002ac2: 44 38 f0                    	cmpb	%r14b, %al
100002ac5: 44 0f 4f eb                 	cmovgl	%ebx, %r13d
100002ac9: 45 0f b6 f6                 	movzbl	%r14b, %r14d
100002acd: 44 0f 4d f0                 	cmovgel	%eax, %r14d
100002ad1: ff c3                       	incl	%ebx
100002ad3: 41 3b 5c 24 14              	cmpl	20(%r12), %ebx
100002ad8: 72 d6                       	jb	-42 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x450>
100002ada: 41 01 df                    	addl	%ebx, %r15d
100002add: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002ae1: e9 70 ff ff ff              	jmp	-144 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3f6>
100002ae6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002af0: 31 c9                       	xorl	%ecx, %ecx
100002af2: ff c2                       	incl	%edx
100002af4: 39 c2                       	cmpl	%eax, %edx
100002af6: 0f 82 26 ff ff ff           	jb	-218 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3c2>
100002afc: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002b03: 48 85 c0                    	testq	%rax, %rax
100002b06: 74 12                       	je	18 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4ba>
100002b08: f0                          	lock
100002b09: ff 48 14                    	decl	20(%rax)
100002b0c: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4ba>
100002b0e: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002b15: e8 dc 42 00 00              	callq	17116 <dyld_stub_binder+0x100006df6>
100002b1a: 48 c7 85 68 ff ff ff 00 00 00 00    	movq	$0, -152(%rbp)
100002b25: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002b29: c5 fc 11 85 40 ff ff ff     	vmovups	%ymm0, -192(%rbp)
100002b31: 83 bd 34 ff ff ff 00        	cmpl	$0, -204(%rbp)
100002b38: 7e 2c                       	jle	44 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x506>
100002b3a: 48 8b 85 70 ff ff ff        	movq	-144(%rbp), %rax
100002b41: 31 c9                       	xorl	%ecx, %ecx
100002b43: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002b4d: 0f 1f 00                    	nopl	(%rax)
100002b50: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002b57: 48 ff c1                    	incq	%rcx
100002b5a: 48 63 95 34 ff ff ff        	movslq	-204(%rbp), %rdx
100002b61: 48 39 d1                    	cmpq	%rdx, %rcx
100002b64: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4f0>
100002b66: 48 8b bd 78 ff ff ff        	movq	-136(%rbp), %rdi
100002b6d: 48 8d 45 80                 	leaq	-128(%rbp), %rax
100002b71: 48 39 c7                    	cmpq	%rax, %rdi
100002b74: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x51e>
100002b76: c5 f8 77                    	vzeroupper
100002b79: e8 ae 42 00 00              	callq	17070 <dyld_stub_binder+0x100006e2c>
100002b7e: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002b85: 48 85 c0                    	testq	%rax, %rax
100002b88: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x53f>
100002b8a: f0                          	lock
100002b8b: ff 48 14                    	decl	20(%rax)
100002b8e: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x53f>
100002b90: 48 8d bd d0 fe ff ff        	leaq	-304(%rbp), %rdi
100002b97: c5 f8 77                    	vzeroupper
100002b9a: e8 57 42 00 00              	callq	16983 <dyld_stub_binder+0x100006df6>
100002b9f: 48 c7 85 08 ff ff ff 00 00 00 00    	movq	$0, -248(%rbp)
100002baa: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002bae: c5 fc 11 85 e0 fe ff ff     	vmovups	%ymm0, -288(%rbp)
100002bb6: 83 bd d4 fe ff ff 00        	cmpl	$0, -300(%rbp)
100002bbd: 7e 27                       	jle	39 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x586>
100002bbf: 48 8b 85 10 ff ff ff        	movq	-240(%rbp), %rax
100002bc6: 31 c9                       	xorl	%ecx, %ecx
100002bc8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002bd0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002bd7: 48 ff c1                    	incq	%rcx
100002bda: 48 63 95 d4 fe ff ff        	movslq	-300(%rbp), %rdx
100002be1: 48 39 d1                    	cmpq	%rdx, %rcx
100002be4: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x570>
100002be6: 48 8b bd 18 ff ff ff        	movq	-232(%rbp), %rdi
100002bed: 48 8d 85 20 ff ff ff        	leaq	-224(%rbp), %rax
100002bf4: 48 39 c7                    	cmpq	%rax, %rdi
100002bf7: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5a1>
100002bf9: c5 f8 77                    	vzeroupper
100002bfc: e8 2b 42 00 00              	callq	16939 <dyld_stub_binder+0x100006e2c>
100002c01: 48 8b 05 50 64 00 00        	movq	25680(%rip), %rax
100002c08: 48 8b 00                    	movq	(%rax), %rax
100002c0b: 48 3b 45 d0                 	cmpq	-48(%rbp), %rax
100002c0f: 75 18                       	jne	24 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5c9>
100002c11: 48 89 d8                    	movq	%rbx, %rax
100002c14: 48 81 c4 28 01 00 00        	addq	$296, %rsp
100002c1b: 5b                          	popq	%rbx
100002c1c: 41 5c                       	popq	%r12
100002c1e: 41 5d                       	popq	%r13
100002c20: 41 5e                       	popq	%r14
100002c22: 41 5f                       	popq	%r15
100002c24: 5d                          	popq	%rbp
100002c25: c5 f8 77                    	vzeroupper
100002c28: c3                          	retq
100002c29: c5 f8 77                    	vzeroupper
100002c2c: e8 7f 42 00 00              	callq	17023 <dyld_stub_binder+0x100006eb0>
100002c31: 48 89 c7                    	movq	%rax, %rdi
100002c34: e8 27 17 00 00              	callq	5927 <_main+0x1550>
100002c39: 48 89 c7                    	movq	%rax, %rdi
100002c3c: e8 1f 17 00 00              	callq	5919 <_main+0x1550>
100002c41: eb 1e                       	jmp	30 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x601>
100002c43: eb 00                       	jmp	0 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5e5>
100002c45: 49 89 c6                    	movq	%rax, %r14
100002c48: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002c4f: 48 85 c0                    	testq	%rax, %rax
100002c52: 0f 85 0f 01 00 00           	jne	271 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x707>
100002c58: e9 1f 01 00 00              	jmp	287 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002c5d: eb 02                       	jmp	2 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x601>
100002c5f: eb 14                       	jmp	20 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x615>
100002c61: 49 89 c6                    	movq	%rax, %r14
100002c64: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002c6b: 48 85 c0                    	testq	%rax, %rax
100002c6e: 75 7f                       	jne	127 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x68f>
100002c70: e9 8f 00 00 00              	jmp	143 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002c75: 49 89 c6                    	movq	%rax, %r14
100002c78: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002c7c: 48 8b 43 38                 	movq	56(%rbx), %rax
100002c80: 48 85 c0                    	testq	%rax, %rax
100002c83: 74 0e                       	je	14 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x633>
100002c85: f0                          	lock
100002c86: ff 48 14                    	decl	20(%rax)
100002c89: 75 08                       	jne	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x633>
100002c8b: 48 89 df                    	movq	%rbx, %rdi
100002c8e: e8 63 41 00 00              	callq	16739 <dyld_stub_binder+0x100006df6>
100002c93: 48 c7 43 38 00 00 00 00     	movq	$0, 56(%rbx)
100002c9b: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002c9f: c5 fc 11 43 10              	vmovups	%ymm0, 16(%rbx)
100002ca4: 83 7b 04 00                 	cmpl	$0, 4(%rbx)
100002ca8: 7e 20                       	jle	32 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x66a>
100002caa: 48 8b 4d b0                 	movq	-80(%rbp), %rcx
100002cae: 48 8d 41 04                 	leaq	4(%rcx), %rax
100002cb2: 48 8b 49 40                 	movq	64(%rcx), %rcx
100002cb6: 31 d2                       	xorl	%edx, %edx
100002cb8: c7 04 91 00 00 00 00        	movl	$0, (%rcx,%rdx,4)
100002cbf: 48 ff c2                    	incq	%rdx
100002cc2: 48 63 30                    	movslq	(%rax), %rsi
100002cc5: 48 39 f2                    	cmpq	%rsi, %rdx
100002cc8: 7c ee                       	jl	-18 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x658>
100002cca: 48 8b 45 b0                 	movq	-80(%rbp), %rax
100002cce: 48 8b 78 48                 	movq	72(%rax), %rdi
100002cd2: 48 3b bd c8 fe ff ff        	cmpq	-312(%rbp), %rdi
100002cd9: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x683>
100002cdb: c5 f8 77                    	vzeroupper
100002cde: e8 49 41 00 00              	callq	16713 <dyld_stub_binder+0x100006e2c>
100002ce3: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002cea: 48 85 c0                    	testq	%rax, %rax
100002ced: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002cef: f0                          	lock
100002cf0: ff 48 14                    	decl	20(%rax)
100002cf3: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002cf5: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002cfc: c5 f8 77                    	vzeroupper
100002cff: e8 f2 40 00 00              	callq	16626 <dyld_stub_binder+0x100006df6>
100002d04: 48 c7 85 68 ff ff ff 00 00 00 00    	movq	$0, -152(%rbp)
100002d0f: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002d13: c5 fc 11 85 40 ff ff ff     	vmovups	%ymm0, -192(%rbp)
100002d1b: 83 bd 34 ff ff ff 00        	cmpl	$0, -204(%rbp)
100002d22: 7e 1f                       	jle	31 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6e3>
100002d24: 48 8b 85 70 ff ff ff        	movq	-144(%rbp), %rax
100002d2b: 31 c9                       	xorl	%ecx, %ecx
100002d2d: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002d34: 48 ff c1                    	incq	%rcx
100002d37: 48 63 95 34 ff ff ff        	movslq	-204(%rbp), %rdx
100002d3e: 48 39 d1                    	cmpq	%rdx, %rcx
100002d41: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6cd>
100002d43: 48 8b bd 78 ff ff ff        	movq	-136(%rbp), %rdi
100002d4a: 48 8d 45 80                 	leaq	-128(%rbp), %rax
100002d4e: 48 39 c7                    	cmpq	%rax, %rdi
100002d51: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6fb>
100002d53: c5 f8 77                    	vzeroupper
100002d56: e8 d1 40 00 00              	callq	16593 <dyld_stub_binder+0x100006e2c>
100002d5b: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002d62: 48 85 c0                    	testq	%rax, %rax
100002d65: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002d67: f0                          	lock
100002d68: ff 48 14                    	decl	20(%rax)
100002d6b: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002d6d: 48 8d bd d0 fe ff ff        	leaq	-304(%rbp), %rdi
100002d74: c5 f8 77                    	vzeroupper
100002d77: e8 7a 40 00 00              	callq	16506 <dyld_stub_binder+0x100006df6>
100002d7c: 48 c7 85 08 ff ff ff 00 00 00 00    	movq	$0, -248(%rbp)
100002d87: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002d8b: c5 fc 11 85 e0 fe ff ff     	vmovups	%ymm0, -288(%rbp)
100002d93: 83 bd d4 fe ff ff 00        	cmpl	$0, -300(%rbp)
100002d9a: 7e 2a                       	jle	42 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x766>
100002d9c: 48 8b 85 10 ff ff ff        	movq	-240(%rbp), %rax
100002da3: 31 c9                       	xorl	%ecx, %ecx
100002da5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002daf: 90                          	nop
100002db0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002db7: 48 ff c1                    	incq	%rcx
100002dba: 48 63 95 d4 fe ff ff        	movslq	-300(%rbp), %rdx
100002dc1: 48 39 d1                    	cmpq	%rdx, %rcx
100002dc4: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x750>
100002dc6: 48 8b bd 18 ff ff ff        	movq	-232(%rbp), %rdi
100002dcd: 48 8d 85 20 ff ff ff        	leaq	-224(%rbp), %rax
100002dd4: 48 39 c7                    	cmpq	%rax, %rdi
100002dd7: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x781>
100002dd9: c5 f8 77                    	vzeroupper
100002ddc: e8 4b 40 00 00              	callq	16459 <dyld_stub_binder+0x100006e2c>
100002de1: 4c 89 f7                    	movq	%r14, %rdi
100002de4: c5 f8 77                    	vzeroupper
100002de7: e8 ec 3f 00 00              	callq	16364 <dyld_stub_binder+0x100006dd8>
100002dec: 0f 0b                       	ud2
100002dee: 48 89 c7                    	movq	%rax, %rdi
100002df1: e8 6a 15 00 00              	callq	5482 <_main+0x1550>
100002df6: 48 89 c7                    	movq	%rax, %rdi
100002df9: e8 62 15 00 00              	callq	5474 <_main+0x1550>
100002dfe: 48 89 c7                    	movq	%rax, %rdi
100002e01: e8 5a 15 00 00              	callq	5466 <_main+0x1550>
100002e06: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

0000000100002e10 _main:
100002e10: 55                          	pushq	%rbp
100002e11: 48 89 e5                    	movq	%rsp, %rbp
100002e14: 41 57                       	pushq	%r15
100002e16: 41 56                       	pushq	%r14
100002e18: 41 55                       	pushq	%r13
100002e1a: 41 54                       	pushq	%r12
100002e1c: 53                          	pushq	%rbx
100002e1d: 48 83 e4 e0                 	andq	$-32, %rsp
100002e21: 48 81 ec 00 04 00 00        	subq	$1024, %rsp
100002e28: 48 8b 05 29 62 00 00        	movq	25129(%rip), %rax
100002e2f: 48 8b 00                    	movq	(%rax), %rax
100002e32: 48 89 84 24 e0 03 00 00     	movq	%rax, 992(%rsp)
100002e3a: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100002e42: e8 f9 27 00 00              	callq	10233 <__ZN11LineNetworkC1Ev>
100002e47: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002e4b: c5 f9 7f 84 24 60 02 00 00  	vmovdqa	%xmm0, 608(%rsp)
100002e54: 48 c7 84 24 70 02 00 00 00 00 00 00 	movq	$0, 624(%rsp)
100002e60: bf 30 00 00 00              	movl	$48, %edi
100002e65: e8 34 40 00 00              	callq	16436 <dyld_stub_binder+0x100006e9e>
100002e6a: 48 89 84 24 70 02 00 00     	movq	%rax, 624(%rsp)
100002e72: c5 f8 28 05 36 42 00 00     	vmovaps	16950(%rip), %xmm0
100002e7a: c5 f8 29 84 24 60 02 00 00  	vmovaps	%xmm0, 608(%rsp)
100002e83: c5 fe 6f 05 2d 60 00 00     	vmovdqu	24621(%rip), %ymm0
100002e8b: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002e8f: 48 b9 69 64 65 6f 2e 6d 70 34       	movabsq	$3778640133568685161, %rcx
100002e99: 48 89 48 20                 	movq	%rcx, 32(%rax)
100002e9d: c6 40 28 00                 	movb	$0, 40(%rax)
100002ea1: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100002ea9: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
100002eb1: 31 d2                       	xorl	%edx, %edx
100002eb3: c5 f8 77                    	vzeroupper
100002eb6: e8 29 3f 00 00              	callq	16169 <dyld_stub_binder+0x100006de4>
100002ebb: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100002ec3: 74 0d                       	je	13 <_main+0xc2>
100002ec5: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
100002ecd: e8 c0 3f 00 00              	callq	16320 <dyld_stub_binder+0x100006e92>
100002ed2: 4c 8d 6c 24 68              	leaq	104(%rsp), %r13
100002ed7: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002edb: c5 f9 d6 44 24 78           	vmovq	%xmm0, 120(%rsp)
100002ee1: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100002ee9: 4c 8d b4 24 c0 03 00 00     	leaq	960(%rsp), %r14
100002ef1: 4c 8d a4 24 c0 01 00 00     	leaq	448(%rsp), %r12
100002ef9: eb 0e                       	jmp	14 <_main+0xf9>
100002efb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100002f00: 45 85 ff                    	testl	%r15d, %r15d
100002f03: 0f 85 d1 0f 00 00           	jne	4049 <_main+0x10ca>
100002f09: 48 89 df                    	movq	%rbx, %rdi
100002f0c: c5 f8 77                    	vzeroupper
100002f0f: e8 24 3f 00 00              	callq	16164 <dyld_stub_binder+0x100006e38>
100002f14: 84 c0                       	testb	%al, %al
100002f16: 0f 84 be 0f 00 00           	je	4030 <_main+0x10ca>
100002f1c: c7 44 24 18 00 00 ff 42     	movl	$1124007936, 24(%rsp)
100002f24: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002f28: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
100002f2d: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
100002f32: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002f36: 48 8d 44 24 20              	leaq	32(%rsp), %rax
100002f3b: 48 89 44 24 58              	movq	%rax, 88(%rsp)
100002f40: 4c 89 6c 24 60              	movq	%r13, 96(%rsp)
100002f45: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002f49: c4 c1 7a 7f 45 00           	vmovdqu	%xmm0, (%r13)
100002f4f: 48 89 df                    	movq	%rbx, %rdi
100002f52: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100002f57: c5 f8 77                    	vzeroupper
100002f5a: e8 91 3e 00 00              	callq	16017 <dyld_stub_binder+0x100006df0>
100002f5f: 41 bf 03 00 00 00           	movl	$3, %r15d
100002f65: 48 83 7c 24 28 00           	cmpq	$0, 40(%rsp)
100002f6b: 0f 84 8f 08 00 00           	je	2191 <_main+0x9f0>
100002f71: 8b 44 24 1c                 	movl	28(%rsp), %eax
100002f75: 83 f8 03                    	cmpl	$3, %eax
100002f78: 0f 8d 62 03 00 00           	jge	866 <_main+0x4d0>
100002f7e: 48 63 4c 24 20              	movslq	32(%rsp), %rcx
100002f83: 48 63 74 24 24              	movslq	36(%rsp), %rsi
100002f88: 48 0f af f1                 	imulq	%rcx, %rsi
100002f8c: 85 c0                       	testl	%eax, %eax
100002f8e: 0f 84 6c 08 00 00           	je	2156 <_main+0x9f0>
100002f94: 48 85 f6                    	testq	%rsi, %rsi
100002f97: 0f 84 63 08 00 00           	je	2147 <_main+0x9f0>
100002f9d: bf 19 00 00 00              	movl	$25, %edi
100002fa2: c5 f8 77                    	vzeroupper
100002fa5: e8 76 3e 00 00              	callq	15990 <dyld_stub_binder+0x100006e20>
100002faa: 3c 1b                       	cmpb	$27, %al
100002fac: 0f 84 4e 08 00 00           	je	2126 <_main+0x9f0>
100002fb2: e8 b1 3e 00 00              	callq	16049 <dyld_stub_binder+0x100006e68>
100002fb7: 49 89 c5                    	movq	%rax, %r13
100002fba: 48 8d 9c 24 e0 00 00 00     	leaq	224(%rsp), %rbx
100002fc2: 48 89 df                    	movq	%rbx, %rdi
100002fc5: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100002fca: 48 8d 94 24 08 02 00 00     	leaq	520(%rsp), %rdx
100002fd2: c5 f9 6e 05 b2 40 00 00     	vmovd	16562(%rip), %xmm0
100002fda: e8 81 f6 ff ff              	callq	-2431 <__Z14get_predictionRN2cv3MatER14ModelInterfacef>
100002fdf: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100002fe7: c5 fa 7e 05 61 40 00 00     	vmovq	16481(%rip), %xmm0
100002fef: 48 89 de                    	movq	%rbx, %rsi
100002ff2: e8 3b 3e 00 00              	callq	15931 <dyld_stub_binder+0x100006e32>
100002ff7: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
100002fff: 48 85 c0                    	testq	%rax, %rax
100003002: 74 0e                       	je	14 <_main+0x202>
100003004: f0                          	lock
100003005: ff 48 14                    	decl	20(%rax)
100003008: 75 08                       	jne	8 <_main+0x202>
10000300a: 48 89 df                    	movq	%rbx, %rdi
10000300d: e8 e4 3d 00 00              	callq	15844 <dyld_stub_binder+0x100006df6>
100003012: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
10000301e: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100003026: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000302a: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
10000302e: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100003036: 7e 2f                       	jle	47 <_main+0x257>
100003038: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100003040: 31 c9                       	xorl	%ecx, %ecx
100003042: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000304c: 0f 1f 40 00                 	nopl	(%rax)
100003050: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003057: 48 ff c1                    	incq	%rcx
10000305a: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
100003062: 48 39 d1                    	cmpq	%rdx, %rcx
100003065: 7c e9                       	jl	-23 <_main+0x240>
100003067: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
10000306f: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
100003077: 48 39 c7                    	cmpq	%rax, %rdi
10000307a: 74 08                       	je	8 <_main+0x274>
10000307c: c5 f8 77                    	vzeroupper
10000307f: e8 a8 3d 00 00              	callq	15784 <dyld_stub_binder+0x100006e2c>
100003084: c5 f8 77                    	vzeroupper
100003087: e8 dc 3d 00 00              	callq	15836 <dyld_stub_binder+0x100006e68>
10000308c: 49 89 c7                    	movq	%rax, %r15
10000308f: c7 84 24 e0 00 00 00 00 00 ff 42    	movl	$1124007936, 224(%rsp)
10000309a: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
1000030a2: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000030a6: c5 fe 7f 40 f4              	vmovdqu	%ymm0, -12(%rax)
1000030ab: c5 fe 7f 40 10              	vmovdqu	%ymm0, 16(%rax)
1000030b0: 48 8b 44 24 20              	movq	32(%rsp), %rax
1000030b5: 48 8d 8c 24 e8 00 00 00     	leaq	232(%rsp), %rcx
1000030bd: 48 89 8c 24 20 01 00 00     	movq	%rcx, 288(%rsp)
1000030c5: 48 8d 8c 24 30 01 00 00     	leaq	304(%rsp), %rcx
1000030cd: 48 89 8c 24 28 01 00 00     	movq	%rcx, 296(%rsp)
1000030d5: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000030d9: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
1000030dd: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
1000030e5: 48 89 df                    	movq	%rbx, %rdi
1000030e8: be 02 00 00 00              	movl	$2, %esi
1000030ed: 4c 89 f2                    	movq	%r14, %rdx
1000030f0: 31 c9                       	xorl	%ecx, %ecx
1000030f2: c5 f8 77                    	vzeroupper
1000030f5: e8 02 3d 00 00              	callq	15618 <dyld_stub_binder+0x100006dfc>
1000030fa: 48 8d 9c 24 80 00 00 00     	leaq	128(%rsp), %rbx
100003102: 48 89 df                    	movq	%rbx, %rdi
100003105: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
10000310d: e8 cc 3c 00 00              	callq	15564 <dyld_stub_binder+0x100006dde>
100003112: 48 c7 84 24 70 01 00 00 00 00 00 00 	movq	$0, 368(%rsp)
10000311e: c7 84 24 60 01 00 00 00 00 01 02    	movl	$33619968, 352(%rsp)
100003129: 48 8d 84 24 e0 00 00 00     	leaq	224(%rsp), %rax
100003131: 48 89 84 24 68 01 00 00     	movq	%rax, 360(%rsp)
100003139: 8b 44 24 20                 	movl	32(%rsp), %eax
10000313d: 8b 4c 24 24                 	movl	36(%rsp), %ecx
100003141: 89 8c 24 b0 01 00 00        	movl	%ecx, 432(%rsp)
100003148: 89 84 24 b4 01 00 00        	movl	%eax, 436(%rsp)
10000314f: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003153: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
100003157: 48 89 df                    	movq	%rbx, %rdi
10000315a: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003162: 48 8d 94 24 b0 01 00 00     	leaq	432(%rsp), %rdx
10000316a: b9 01 00 00 00              	movl	$1, %ecx
10000316f: e8 a0 3c 00 00              	callq	15520 <dyld_stub_binder+0x100006e14>
100003174: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003178: c5 fd 7f 84 24 60 01 00 00  	vmovdqa	%ymm0, 352(%rsp)
100003181: c7 84 24 80 00 00 00 00 00 ff 42    	movl	$1124007936, 128(%rsp)
10000318c: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
100003194: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
100003199: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
10000319d: 48 8b 44 24 20              	movq	32(%rsp), %rax
1000031a2: 48 8d 8c 24 88 00 00 00     	leaq	136(%rsp), %rcx
1000031aa: 48 89 8c 24 c0 00 00 00     	movq	%rcx, 192(%rsp)
1000031b2: 48 8d 8c 24 d0 00 00 00     	leaq	208(%rsp), %rcx
1000031ba: 48 89 8c 24 c8 00 00 00     	movq	%rcx, 200(%rsp)
1000031c2: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000031c6: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
1000031ca: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
1000031d2: 48 89 df                    	movq	%rbx, %rdi
1000031d5: be 02 00 00 00              	movl	$2, %esi
1000031da: 4c 89 f2                    	movq	%r14, %rdx
1000031dd: b9 10 00 00 00              	movl	$16, %ecx
1000031e2: c5 f8 77                    	vzeroupper
1000031e5: e8 12 3c 00 00              	callq	15378 <dyld_stub_binder+0x100006dfc>
1000031ea: 48 89 df                    	movq	%rbx, %rdi
1000031ed: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
1000031f5: e8 0e 3c 00 00              	callq	15374 <dyld_stub_binder+0x100006e08>
1000031fa: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000031ff: 48 85 c0                    	testq	%rax, %rax
100003202: 74 04                       	je	4 <_main+0x3f8>
100003204: f0                          	lock
100003205: ff 40 14                    	incl	20(%rax)
100003208: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003210: 48 85 c0                    	testq	%rax, %rax
100003213: 74 13                       	je	19 <_main+0x418>
100003215: f0                          	lock
100003216: ff 48 14                    	decl	20(%rax)
100003219: 75 0d                       	jne	13 <_main+0x418>
10000321b: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
100003223: e8 ce 3b 00 00              	callq	15310 <dyld_stub_binder+0x100006df6>
100003228: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
100003234: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
10000323c: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003240: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100003245: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
10000324d: 0f 8e 2c 06 00 00           	jle	1580 <_main+0xa6f>
100003253: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
10000325b: 31 c9                       	xorl	%ecx, %ecx
10000325d: 0f 1f 00                    	nopl	(%rax)
100003260: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003267: 48 ff c1                    	incq	%rcx
10000326a: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
100003272: 48 39 d1                    	cmpq	%rdx, %rcx
100003275: 7c e9                       	jl	-23 <_main+0x450>
100003277: 8b 44 24 18                 	movl	24(%rsp), %eax
10000327b: 89 84 24 80 00 00 00        	movl	%eax, 128(%rsp)
100003282: 83 fa 02                    	cmpl	$2, %edx
100003285: 0f 8f 0c 06 00 00           	jg	1548 <_main+0xa87>
10000328b: 8b 44 24 1c                 	movl	28(%rsp), %eax
10000328f: 83 f8 02                    	cmpl	$2, %eax
100003292: 0f 8f ff 05 00 00           	jg	1535 <_main+0xa87>
100003298: 89 84 24 84 00 00 00        	movl	%eax, 132(%rsp)
10000329f: 8b 4c 24 20                 	movl	32(%rsp), %ecx
1000032a3: 8b 44 24 24                 	movl	36(%rsp), %eax
1000032a7: 89 8c 24 88 00 00 00        	movl	%ecx, 136(%rsp)
1000032ae: 89 84 24 8c 00 00 00        	movl	%eax, 140(%rsp)
1000032b5: 48 8b 44 24 60              	movq	96(%rsp), %rax
1000032ba: 48 8b 10                    	movq	(%rax), %rdx
1000032bd: 48 8b b4 24 c8 00 00 00     	movq	200(%rsp), %rsi
1000032c5: 48 89 16                    	movq	%rdx, (%rsi)
1000032c8: 48 8b 40 08                 	movq	8(%rax), %rax
1000032cc: 48 89 46 08                 	movq	%rax, 8(%rsi)
1000032d0: e9 db 05 00 00              	jmp	1499 <_main+0xaa0>
1000032d5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000032df: 90                          	nop
1000032e0: 48 8b 4c 24 58              	movq	88(%rsp), %rcx
1000032e5: 83 f8 0f                    	cmpl	$15, %eax
1000032e8: 77 0c                       	ja	12 <_main+0x4e6>
1000032ea: be 01 00 00 00              	movl	$1, %esi
1000032ef: 31 d2                       	xorl	%edx, %edx
1000032f1: e9 ea 04 00 00              	jmp	1258 <_main+0x9d0>
1000032f6: 89 c2                       	movl	%eax, %edx
1000032f8: 83 e2 f0                    	andl	$-16, %edx
1000032fb: 48 8d 72 f0                 	leaq	-16(%rdx), %rsi
1000032ff: 48 89 f7                    	movq	%rsi, %rdi
100003302: 48 c1 ef 04                 	shrq	$4, %rdi
100003306: 48 ff c7                    	incq	%rdi
100003309: 89 fb                       	movl	%edi, %ebx
10000330b: 83 e3 03                    	andl	$3, %ebx
10000330e: 48 83 fe 30                 	cmpq	$48, %rsi
100003312: 73 25                       	jae	37 <_main+0x529>
100003314: c4 e2 7d 59 05 2b 3d 00 00  	vpbroadcastq	15659(%rip), %ymm0
10000331d: 31 ff                       	xorl	%edi, %edi
10000331f: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
100003323: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100003327: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
10000332b: 48 85 db                    	testq	%rbx, %rbx
10000332e: 0f 85 0e 03 00 00           	jne	782 <_main+0x832>
100003334: e9 d0 03 00 00              	jmp	976 <_main+0x8f9>
100003339: 48 89 de                    	movq	%rbx, %rsi
10000333c: 48 29 fe                    	subq	%rdi, %rsi
10000333f: c4 e2 7d 59 05 00 3d 00 00  	vpbroadcastq	15616(%rip), %ymm0
100003348: 31 ff                       	xorl	%edi, %edi
10000334a: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
10000334e: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100003352: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
100003356: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003360: c4 e2 7d 25 24 b9           	vpmovsxdq	(%rcx,%rdi,4), %ymm4
100003366: c4 e2 7d 25 6c b9 10        	vpmovsxdq	16(%rcx,%rdi,4), %ymm5
10000336d: c4 e2 7d 25 74 b9 20        	vpmovsxdq	32(%rcx,%rdi,4), %ymm6
100003374: c4 e2 7d 25 7c b9 30        	vpmovsxdq	48(%rcx,%rdi,4), %ymm7
10000337b: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
100003380: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
100003384: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
100003389: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
10000338e: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
100003393: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003399: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000339d: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000033a2: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
1000033a7: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
1000033ab: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
1000033b0: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
1000033b5: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
1000033b9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000033be: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000033c2: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000033c6: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
1000033cb: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
1000033cf: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
1000033d4: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
1000033d8: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
1000033dc: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000033e1: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
1000033e5: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000033e9: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
1000033ee: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
1000033f2: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
1000033f7: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
1000033fb: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
1000033ff: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003404: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003408: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000340c: c4 e2 7d 25 64 b9 40        	vpmovsxdq	64(%rcx,%rdi,4), %ymm4
100003413: c4 e2 7d 25 6c b9 50        	vpmovsxdq	80(%rcx,%rdi,4), %ymm5
10000341a: c4 e2 7d 25 74 b9 60        	vpmovsxdq	96(%rcx,%rdi,4), %ymm6
100003421: c4 e2 7d 25 7c b9 70        	vpmovsxdq	112(%rcx,%rdi,4), %ymm7
100003428: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
10000342d: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100003432: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100003437: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
10000343b: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100003440: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003446: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000344a: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
10000344f: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100003454: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100003458: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
10000345d: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100003461: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100003466: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000346b: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
10000346f: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100003473: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100003478: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
10000347c: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100003481: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100003485: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100003489: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000348e: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003492: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003496: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
10000349b: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
10000349f: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
1000034a4: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
1000034a8: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000034ac: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000034b1: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
1000034b5: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000034b9: c4 e2 7d 25 a4 b9 80 00 00 00       	vpmovsxdq	128(%rcx,%rdi,4), %ymm4
1000034c3: c4 e2 7d 25 ac b9 90 00 00 00       	vpmovsxdq	144(%rcx,%rdi,4), %ymm5
1000034cd: c4 e2 7d 25 b4 b9 a0 00 00 00       	vpmovsxdq	160(%rcx,%rdi,4), %ymm6
1000034d7: c4 e2 7d 25 bc b9 b0 00 00 00       	vpmovsxdq	176(%rcx,%rdi,4), %ymm7
1000034e1: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
1000034e6: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
1000034eb: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
1000034f0: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
1000034f4: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
1000034f9: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000034ff: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100003503: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003508: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
10000350d: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100003511: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
100003516: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
10000351a: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
10000351f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003524: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003528: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
10000352c: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100003531: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100003535: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
10000353a: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
10000353e: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100003542: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003547: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
10000354b: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
10000354f: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
100003554: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100003558: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
10000355d: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100003561: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100003565: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000356a: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
10000356e: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100003572: c4 e2 7d 25 a4 b9 c0 00 00 00       	vpmovsxdq	192(%rcx,%rdi,4), %ymm4
10000357c: c4 e2 7d 25 ac b9 d0 00 00 00       	vpmovsxdq	208(%rcx,%rdi,4), %ymm5
100003586: c4 e2 7d 25 b4 b9 e0 00 00 00       	vpmovsxdq	224(%rcx,%rdi,4), %ymm6
100003590: c4 e2 7d 25 bc b9 f0 00 00 00       	vpmovsxdq	240(%rcx,%rdi,4), %ymm7
10000359a: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
10000359f: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
1000035a4: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
1000035a9: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
1000035ad: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
1000035b2: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000035b8: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000035bc: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000035c1: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
1000035c6: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
1000035ca: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
1000035cf: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
1000035d3: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
1000035d8: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000035dd: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000035e1: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000035e5: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
1000035ea: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
1000035ee: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
1000035f3: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
1000035f7: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000035fb: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003600: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003604: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003608: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
10000360d: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100003611: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100003616: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
10000361a: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
10000361e: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003623: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003627: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000362b: 48 83 c7 40                 	addq	$64, %rdi
10000362f: 48 83 c6 04                 	addq	$4, %rsi
100003633: 0f 85 27 fd ff ff           	jne	-729 <_main+0x550>
100003639: 48 85 db                    	testq	%rbx, %rbx
10000363c: 0f 84 c7 00 00 00           	je	199 <_main+0x8f9>
100003642: 48 8d 34 b9                 	leaq	(%rcx,%rdi,4), %rsi
100003646: 48 83 c6 30                 	addq	$48, %rsi
10000364a: 48 c1 e3 06                 	shlq	$6, %rbx
10000364e: 31 ff                       	xorl	%edi, %edi
100003650: c4 e2 7d 25 64 3e d0        	vpmovsxdq	-48(%rsi,%rdi), %ymm4
100003657: c4 e2 7d 25 6c 3e e0        	vpmovsxdq	-32(%rsi,%rdi), %ymm5
10000365e: c4 e2 7d 25 74 3e f0        	vpmovsxdq	-16(%rsi,%rdi), %ymm6
100003665: c4 e2 7d 25 3c 3e           	vpmovsxdq	(%rsi,%rdi), %ymm7
10000366b: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
100003670: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
100003674: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
100003679: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
10000367e: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
100003683: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003689: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000368d: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003692: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
100003697: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
10000369b: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
1000036a0: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
1000036a5: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
1000036a9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000036ae: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000036b2: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000036b6: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
1000036bb: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
1000036bf: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
1000036c4: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
1000036c8: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
1000036cc: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000036d1: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
1000036d5: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000036d9: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
1000036de: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
1000036e2: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
1000036e7: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
1000036eb: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
1000036ef: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000036f4: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
1000036f8: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000036fc: 48 83 c7 40                 	addq	$64, %rdi
100003700: 48 39 fb                    	cmpq	%rdi, %rbx
100003703: 0f 85 47 ff ff ff           	jne	-185 <_main+0x840>
100003709: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
10000370e: c5 dd f4 e0                 	vpmuludq	%ymm0, %ymm4, %ymm4
100003712: c5 d5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm5
100003717: c5 e5 f4 ed                 	vpmuludq	%ymm5, %ymm3, %ymm5
10000371b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000371f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003724: c5 e5 f4 c0                 	vpmuludq	%ymm0, %ymm3, %ymm0
100003728: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
10000372c: c5 e5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm3
100003731: c5 e5 f4 d8                 	vpmuludq	%ymm0, %ymm3, %ymm3
100003735: c5 dd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm4
10000373a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
10000373e: c5 dd d4 db                 	vpaddq	%ymm3, %ymm4, %ymm3
100003742: c5 e5 73 f3 20              	vpsllq	$32, %ymm3, %ymm3
100003747: c5 ed f4 c0                 	vpmuludq	%ymm0, %ymm2, %ymm0
10000374b: c5 fd d4 c3                 	vpaddq	%ymm3, %ymm0, %ymm0
10000374f: c5 ed 73 d1 20              	vpsrlq	$32, %ymm1, %ymm2
100003754: c5 ed f4 d0                 	vpmuludq	%ymm0, %ymm2, %ymm2
100003758: c5 e5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm3
10000375d: c5 f5 f4 db                 	vpmuludq	%ymm3, %ymm1, %ymm3
100003761: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
100003765: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
10000376a: c5 f5 f4 c0                 	vpmuludq	%ymm0, %ymm1, %ymm0
10000376e: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
100003772: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100003778: c5 ed 73 d0 20              	vpsrlq	$32, %ymm0, %ymm2
10000377d: c5 ed f4 d1                 	vpmuludq	%ymm1, %ymm2, %ymm2
100003781: c5 e5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm3
100003786: c5 fd f4 db                 	vpmuludq	%ymm3, %ymm0, %ymm3
10000378a: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
10000378e: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
100003793: c5 fd f4 c1                 	vpmuludq	%ymm1, %ymm0, %ymm0
100003797: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
10000379b: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
1000037a0: c5 e9 73 d0 20              	vpsrlq	$32, %xmm0, %xmm2
1000037a5: c5 e9 f4 d1                 	vpmuludq	%xmm1, %xmm2, %xmm2
1000037a9: c5 e1 73 d8 0c              	vpsrldq	$12, %xmm0, %xmm3
1000037ae: c5 f9 f4 db                 	vpmuludq	%xmm3, %xmm0, %xmm3
1000037b2: c5 e1 d4 d2                 	vpaddq	%xmm2, %xmm3, %xmm2
1000037b6: c5 e9 73 f2 20              	vpsllq	$32, %xmm2, %xmm2
1000037bb: c5 f9 f4 c1                 	vpmuludq	%xmm1, %xmm0, %xmm0
1000037bf: c5 f9 d4 c2                 	vpaddq	%xmm2, %xmm0, %xmm0
1000037c3: c4 e1 f9 7e c6              	vmovq	%xmm0, %rsi
1000037c8: 48 39 c2                    	cmpq	%rax, %rdx
1000037cb: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
1000037d3: 74 1b                       	je	27 <_main+0x9e0>
1000037d5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000037df: 90                          	nop
1000037e0: 48 63 3c 91                 	movslq	(%rcx,%rdx,4), %rdi
1000037e4: 48 0f af f7                 	imulq	%rdi, %rsi
1000037e8: 48 ff c2                    	incq	%rdx
1000037eb: 48 39 d0                    	cmpq	%rdx, %rax
1000037ee: 75 f0                       	jne	-16 <_main+0x9d0>
1000037f0: 85 c0                       	testl	%eax, %eax
1000037f2: 0f 85 9c f7 ff ff           	jne	-2148 <_main+0x184>
1000037f8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100003800: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003805: 48 85 c0                    	testq	%rax, %rax
100003808: 74 13                       	je	19 <_main+0xa0d>
10000380a: f0                          	lock
10000380b: ff 48 14                    	decl	20(%rax)
10000380e: 75 0d                       	jne	13 <_main+0xa0d>
100003810: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
100003815: c5 f8 77                    	vzeroupper
100003818: e8 d9 35 00 00              	callq	13785 <dyld_stub_binder+0x100006df6>
10000381d: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
100003826: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000382a: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
10000382f: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100003834: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
100003839: 7e 29                       	jle	41 <_main+0xa54>
10000383b: 48 8b 44 24 58              	movq	88(%rsp), %rax
100003840: 31 c9                       	xorl	%ecx, %ecx
100003842: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000384c: 0f 1f 40 00                 	nopl	(%rax)
100003850: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003857: 48 ff c1                    	incq	%rcx
10000385a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000385f: 48 39 d1                    	cmpq	%rdx, %rcx
100003862: 7c ec                       	jl	-20 <_main+0xa40>
100003864: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003869: 4c 39 ef                    	cmpq	%r13, %rdi
10000386c: 0f 84 8e f6 ff ff           	je	-2418 <_main+0xf0>
100003872: c5 f8 77                    	vzeroupper
100003875: e8 b2 35 00 00              	callq	13746 <dyld_stub_binder+0x100006e2c>
10000387a: e9 81 f6 ff ff              	jmp	-2431 <_main+0xf0>
10000387f: 8b 44 24 18                 	movl	24(%rsp), %eax
100003883: 89 84 24 80 00 00 00        	movl	%eax, 128(%rsp)
10000388a: 8b 44 24 1c                 	movl	28(%rsp), %eax
10000388e: 83 f8 02                    	cmpl	$2, %eax
100003891: 0f 8e 01 fa ff ff           	jle	-1535 <_main+0x488>
100003897: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
10000389f: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
1000038a4: c5 f8 77                    	vzeroupper
1000038a7: e8 56 35 00 00              	callq	13654 <dyld_stub_binder+0x100006e02>
1000038ac: 8b 4c 24 20                 	movl	32(%rsp), %ecx
1000038b0: c4 c1 eb 2a c5              	vcvtsi2sd	%r13, %xmm2, %xmm0
1000038b5: c4 c1 eb 2a cf              	vcvtsi2sd	%r15, %xmm2, %xmm1
1000038ba: c5 fb 10 15 7e 37 00 00     	vmovsd	14206(%rip), %xmm2
1000038c2: c5 fb 5e c2                 	vdivsd	%xmm2, %xmm0, %xmm0
1000038c6: c5 f3 5e ca                 	vdivsd	%xmm2, %xmm1, %xmm1
1000038ca: c5 fc 10 54 24 28           	vmovups	40(%rsp), %ymm2
1000038d0: c5 fc 11 94 24 90 00 00 00  	vmovups	%ymm2, 144(%rsp)
1000038d9: c5 f9 10 54 24 48           	vmovupd	72(%rsp), %xmm2
1000038df: c5 f9 11 94 24 b0 00 00 00  	vmovupd	%xmm2, 176(%rsp)
1000038e8: 85 c9                       	testl	%ecx, %ecx
1000038ea: 4d 89 f5                    	movq	%r14, %r13
1000038ed: 0f 84 53 01 00 00           	je	339 <_main+0xc36>
1000038f3: 31 c0                       	xorl	%eax, %eax
1000038f5: 8b 74 24 24                 	movl	36(%rsp), %esi
1000038f9: 85 f6                       	testl	%esi, %esi
1000038fb: be 00 00 00 00              	movl	$0, %esi
100003900: 75 21                       	jne	33 <_main+0xb13>
100003902: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000390c: 0f 1f 40 00                 	nopl	(%rax)
100003910: ff c0                       	incl	%eax
100003912: 39 c8                       	cmpl	%ecx, %eax
100003914: 0f 83 2c 01 00 00           	jae	300 <_main+0xc36>
10000391a: 85 f6                       	testl	%esi, %esi
10000391c: be 00 00 00 00              	movl	$0, %esi
100003921: 74 ed                       	je	-19 <_main+0xb00>
100003923: 48 63 c8                    	movslq	%eax, %rcx
100003926: 31 d2                       	xorl	%edx, %edx
100003928: c5 fb 10 25 28 37 00 00     	vmovsd	14120(%rip), %xmm4
100003930: c5 fa 10 2d 58 37 00 00     	vmovss	14168(%rip), %xmm5
100003938: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100003940: 48 8b 74 24 60              	movq	96(%rsp), %rsi
100003945: 48 8b 3e                    	movq	(%rsi), %rdi
100003948: 48 0f af f9                 	imulq	%rcx, %rdi
10000394c: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100003951: 48 63 d2                    	movslq	%edx, %rdx
100003954: 48 8d 34 52                 	leaq	(%rdx,%rdx,2), %rsi
100003958: 0f b6 3c 37                 	movzbl	(%rdi,%rsi), %edi
10000395c: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003960: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003964: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003968: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100003970: 48 8b 1b                    	movq	(%rbx), %rbx
100003973: 48 0f af d9                 	imulq	%rcx, %rbx
100003977: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
10000397f: 40 88 3c 33                 	movb	%dil, (%rbx,%rsi)
100003983: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003988: 48 8b 3f                    	movq	(%rdi), %rdi
10000398b: 48 0f af f9                 	imulq	%rcx, %rdi
10000398f: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100003994: 0f b6 7c 37 01              	movzbl	1(%rdi,%rsi), %edi
100003999: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
10000399d: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
1000039a5: 48 8b 3f                    	movq	(%rdi), %rdi
1000039a8: 48 0f af f9                 	imulq	%rcx, %rdi
1000039ac: 48 03 bc 24 f0 00 00 00     	addq	240(%rsp), %rdi
1000039b4: 0f b6 3c 3a                 	movzbl	(%rdx,%rdi), %edi
1000039b8: c5 ca 2a df                 	vcvtsi2ss	%edi, %xmm6, %xmm3
1000039bc: c5 e2 59 dd                 	vmulss	%xmm5, %xmm3, %xmm3
1000039c0: c5 e2 5a db                 	vcvtss2sd	%xmm3, %xmm3, %xmm3
1000039c4: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
1000039c8: c5 eb 58 d3                 	vaddsd	%xmm3, %xmm2, %xmm2
1000039cc: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
1000039d0: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
1000039d8: 48 8b 1b                    	movq	(%rbx), %rbx
1000039db: 48 0f af d9                 	imulq	%rcx, %rbx
1000039df: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
1000039e7: 40 88 7c 33 01              	movb	%dil, 1(%rbx,%rsi)
1000039ec: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
1000039f1: 48 8b 3f                    	movq	(%rdi), %rdi
1000039f4: 48 0f af f9                 	imulq	%rcx, %rdi
1000039f8: 48 03 7c 24 28              	addq	40(%rsp), %rdi
1000039fd: 0f b6 7c 37 02              	movzbl	2(%rdi,%rsi), %edi
100003a02: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003a06: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003a0a: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003a0e: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100003a16: 48 8b 1b                    	movq	(%rbx), %rbx
100003a19: 48 0f af d9                 	imulq	%rcx, %rbx
100003a1d: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100003a25: 40 88 7c 33 02              	movb	%dil, 2(%rbx,%rsi)
100003a2a: ff c2                       	incl	%edx
100003a2c: 8b 74 24 24                 	movl	36(%rsp), %esi
100003a30: 39 f2                       	cmpl	%esi, %edx
100003a32: 0f 82 08 ff ff ff           	jb	-248 <_main+0xb30>
100003a38: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100003a3c: ff c0                       	incl	%eax
100003a3e: 39 c8                       	cmpl	%ecx, %eax
100003a40: 0f 82 d4 fe ff ff           	jb	-300 <_main+0xb0a>
100003a46: c5 fb 10 15 12 36 00 00     	vmovsd	13842(%rip), %xmm2
100003a4e: c5 eb 59 54 24 78           	vmulsd	120(%rsp), %xmm2, %xmm2
100003a54: c5 f3 5c c0                 	vsubsd	%xmm0, %xmm1, %xmm0
100003a58: c5 fb 58 05 08 36 00 00     	vaddsd	13832(%rip), %xmm0, %xmm0
100003a60: c5 fb 10 0d 08 36 00 00     	vmovsd	13832(%rip), %xmm1
100003a68: c5 f3 5e c0                 	vdivsd	%xmm0, %xmm1, %xmm0
100003a6c: c5 eb 58 c0                 	vaddsd	%xmm0, %xmm2, %xmm0
100003a70: 8b 9c 24 28 02 00 00        	movl	552(%rsp), %ebx
100003a77: c5 fb 11 44 24 78           	vmovsd	%xmm0, 120(%rsp)
100003a7d: c5 f8 77                    	vzeroupper
100003a80: e8 37 34 00 00              	callq	13367 <dyld_stub_binder+0x100006ebc>
100003a85: c5 fb 2c f0                 	vcvttsd2si	%xmm0, %esi
100003a89: 4c 89 e7                    	movq	%r12, %rdi
100003a8c: e8 f5 33 00 00              	callq	13301 <dyld_stub_binder+0x100006e86>
100003a91: 4c 89 e7                    	movq	%r12, %rdi
100003a94: 31 f6                       	xorl	%esi, %esi
100003a96: 48 8d 15 44 54 00 00        	leaq	21572(%rip), %rdx
100003a9d: e8 b4 33 00 00              	callq	13236 <dyld_stub_binder+0x100006e56>
100003aa2: 48 8b 48 10                 	movq	16(%rax), %rcx
100003aa6: 48 89 8c 24 50 01 00 00     	movq	%rcx, 336(%rsp)
100003aae: c5 f9 10 00                 	vmovupd	(%rax), %xmm0
100003ab2: c5 f9 29 84 24 40 01 00 00  	vmovapd	%xmm0, 320(%rsp)
100003abb: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003abf: c5 f9 11 00                 	vmovupd	%xmm0, (%rax)
100003ac3: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003acb: 48 8d bc 24 40 01 00 00     	leaq	320(%rsp), %rdi
100003ad3: 48 8d 35 0e 54 00 00        	leaq	21518(%rip), %rsi
100003ada: e8 6b 33 00 00              	callq	13163 <dyld_stub_binder+0x100006e4a>
100003adf: c4 e1 cb 2a c3              	vcvtsi2sd	%rbx, %xmm6, %xmm0
100003ae4: c5 fb 59 44 24 78           	vmulsd	120(%rsp), %xmm0, %xmm0
100003aea: c5 fb 5e 05 86 35 00 00     	vdivsd	13702(%rip), %xmm0, %xmm0
100003af2: 48 8b 48 10                 	movq	16(%rax), %rcx
100003af6: 48 89 8c 24 d0 03 00 00     	movq	%rcx, 976(%rsp)
100003afe: c5 f9 10 08                 	vmovupd	(%rax), %xmm1
100003b02: c5 f9 29 8c 24 c0 03 00 00  	vmovapd	%xmm1, 960(%rsp)
100003b0b: c5 f1 57 c9                 	vxorpd	%xmm1, %xmm1, %xmm1
100003b0f: c5 f9 11 08                 	vmovupd	%xmm1, (%rax)
100003b13: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003b1b: 48 8d bc 24 98 01 00 00     	leaq	408(%rsp), %rdi
100003b23: e8 58 33 00 00              	callq	13144 <dyld_stub_binder+0x100006e80>
100003b28: 0f b6 94 24 98 01 00 00     	movzbl	408(%rsp), %edx
100003b30: f6 c2 01                    	testb	$1, %dl
100003b33: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100003b3b: 74 12                       	je	18 <_main+0xd3f>
100003b3d: 48 8b b4 24 a8 01 00 00     	movq	424(%rsp), %rsi
100003b45: 48 8b 94 24 a0 01 00 00     	movq	416(%rsp), %rdx
100003b4d: eb 0b                       	jmp	11 <_main+0xd4a>
100003b4f: 48 d1 ea                    	shrq	%rdx
100003b52: 48 8d b4 24 99 01 00 00     	leaq	409(%rsp), %rsi
100003b5a: 4c 89 ef                    	movq	%r13, %rdi
100003b5d: e8 ee 32 00 00              	callq	13038 <dyld_stub_binder+0x100006e50>
100003b62: 48 8b 48 10                 	movq	16(%rax), %rcx
100003b66: 48 89 8c 24 70 01 00 00     	movq	%rcx, 368(%rsp)
100003b6e: c5 f8 10 00                 	vmovups	(%rax), %xmm0
100003b72: c5 f8 29 84 24 60 01 00 00  	vmovaps	%xmm0, 352(%rsp)
100003b7b: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003b7f: c5 f8 11 00                 	vmovups	%xmm0, (%rax)
100003b83: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003b8b: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
100003b93: 0f 85 80 01 00 00           	jne	384 <_main+0xf09>
100003b99: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003ba1: 0f 85 8d 01 00 00           	jne	397 <_main+0xf24>
100003ba7: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003baf: 0f 85 9a 01 00 00           	jne	410 <_main+0xf3f>
100003bb5: 4d 89 e7                    	movq	%r12, %r15
100003bb8: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003bc0: 74 0d                       	je	13 <_main+0xdbf>
100003bc2: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
100003bca: e8 c3 32 00 00              	callq	12995 <dyld_stub_binder+0x100006e92>
100003bcf: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
100003bdb: c7 84 24 c0 03 00 00 00 00 01 03    	movl	$50397184, 960(%rsp)
100003be6: 4c 8d a4 24 80 00 00 00     	leaq	128(%rsp), %r12
100003bee: 4c 89 a4 24 c8 03 00 00     	movq	%r12, 968(%rsp)
100003bf6: 48 b8 1e 00 00 00 1e 00 00 00       	movabsq	$128849018910, %rax
100003c00: 48 89 84 24 b8 01 00 00     	movq	%rax, 440(%rsp)
100003c08: c5 fc 28 05 b0 34 00 00     	vmovaps	13488(%rip), %ymm0
100003c10: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100003c19: c7 44 24 08 00 00 00 00     	movl	$0, 8(%rsp)
100003c21: c7 04 24 10 00 00 00        	movl	$16, (%rsp)
100003c28: 4c 89 ef                    	movq	%r13, %rdi
100003c2b: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003c33: 48 8d 94 24 b8 01 00 00     	leaq	440(%rsp), %rdx
100003c3b: 31 c9                       	xorl	%ecx, %ecx
100003c3d: c5 fb 10 05 3b 34 00 00     	vmovsd	13371(%rip), %xmm0
100003c45: 4c 8d 84 24 40 02 00 00     	leaq	576(%rsp), %r8
100003c4d: 41 b9 02 00 00 00           	movl	$2, %r9d
100003c53: c5 f8 77                    	vzeroupper
100003c56: e8 bf 31 00 00              	callq	12735 <dyld_stub_binder+0x100006e1a>
100003c5b: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003c5f: c5 f9 29 84 24 c0 03 00 00  	vmovapd	%xmm0, 960(%rsp)
100003c68: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
100003c74: c6 84 24 c0 03 00 00 0a     	movb	$10, 960(%rsp)
100003c7c: 48 8d 84 24 c1 03 00 00     	leaq	961(%rsp), %rax
100003c84: c6 40 04 65                 	movb	$101, 4(%rax)
100003c88: c7 00 66 72 61 6d           	movl	$1835102822, (%rax)
100003c8e: c6 84 24 c6 03 00 00 00     	movb	$0, 966(%rsp)
100003c96: 48 c7 84 24 50 01 00 00 00 00 00 00 	movq	$0, 336(%rsp)
100003ca2: c7 84 24 40 01 00 00 00 00 01 01    	movl	$16842752, 320(%rsp)
100003cad: 4c 89 a4 24 48 01 00 00     	movq	%r12, 328(%rsp)
100003cb5: 4c 89 ef                    	movq	%r13, %rdi
100003cb8: 48 8d b4 24 40 01 00 00     	leaq	320(%rsp), %rsi
100003cc0: e8 49 31 00 00              	callq	12617 <dyld_stub_binder+0x100006e0e>
100003cc5: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003ccd: 4d 89 fc                    	movq	%r15, %r12
100003cd0: 4c 8d 6c 24 68              	leaq	104(%rsp), %r13
100003cd5: 0f 85 97 00 00 00           	jne	151 <_main+0xf62>
100003cdb: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003ce3: 4c 8d bc 24 80 00 00 00     	leaq	128(%rsp), %r15
100003ceb: 0f 85 a4 00 00 00           	jne	164 <_main+0xf85>
100003cf1: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003cf9: 48 85 c0                    	testq	%rax, %rax
100003cfc: 0f 84 b1 00 00 00           	je	177 <_main+0xfa3>
100003d02: f0                          	lock
100003d03: ff 48 14                    	decl	20(%rax)
100003d06: 0f 85 a7 00 00 00           	jne	167 <_main+0xfa3>
100003d0c: 4c 89 ff                    	movq	%r15, %rdi
100003d0f: e8 e2 30 00 00              	callq	12514 <dyld_stub_binder+0x100006df6>
100003d14: e9 9a 00 00 00              	jmp	154 <_main+0xfa3>
100003d19: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100003d21: e8 6c 31 00 00              	callq	12652 <dyld_stub_binder+0x100006e92>
100003d26: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003d2e: 0f 84 73 fe ff ff           	je	-397 <_main+0xd97>
100003d34: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003d3c: e8 51 31 00 00              	callq	12625 <dyld_stub_binder+0x100006e92>
100003d41: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003d49: 0f 84 66 fe ff ff           	je	-410 <_main+0xda5>
100003d4f: 48 8b bc 24 50 01 00 00     	movq	336(%rsp), %rdi
100003d57: e8 36 31 00 00              	callq	12598 <dyld_stub_binder+0x100006e92>
100003d5c: 4d 89 e7                    	movq	%r12, %r15
100003d5f: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003d67: 0f 85 55 fe ff ff           	jne	-427 <_main+0xdb2>
100003d6d: e9 5d fe ff ff              	jmp	-419 <_main+0xdbf>
100003d72: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003d7a: e8 13 31 00 00              	callq	12563 <dyld_stub_binder+0x100006e92>
100003d7f: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003d87: 4c 8d bc 24 80 00 00 00     	leaq	128(%rsp), %r15
100003d8f: 0f 84 5c ff ff ff           	je	-164 <_main+0xee1>
100003d95: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
100003d9d: e8 f0 30 00 00              	callq	12528 <dyld_stub_binder+0x100006e92>
100003da2: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003daa: 48 85 c0                    	testq	%rax, %rax
100003dad: 0f 85 4f ff ff ff           	jne	-177 <_main+0xef2>
100003db3: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
100003dbf: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
100003dc7: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003dcb: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100003dd0: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
100003dd8: 7e 2d                       	jle	45 <_main+0xff7>
100003dda: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
100003de2: 31 c9                       	xorl	%ecx, %ecx
100003de4: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003dee: 66 90                       	nop
100003df0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003df7: 48 ff c1                    	incq	%rcx
100003dfa: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
100003e02: 48 39 d1                    	cmpq	%rdx, %rcx
100003e05: 7c e9                       	jl	-23 <_main+0xfe0>
100003e07: 48 8b bc 24 c8 00 00 00     	movq	200(%rsp), %rdi
100003e0f: 48 8d 84 24 d0 00 00 00     	leaq	208(%rsp), %rax
100003e17: 48 39 c7                    	cmpq	%rax, %rdi
100003e1a: 74 08                       	je	8 <_main+0x1014>
100003e1c: c5 f8 77                    	vzeroupper
100003e1f: e8 08 30 00 00              	callq	12296 <dyld_stub_binder+0x100006e2c>
100003e24: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
100003e2c: 48 85 c0                    	testq	%rax, %rax
100003e2f: 74 16                       	je	22 <_main+0x1037>
100003e31: f0                          	lock
100003e32: ff 48 14                    	decl	20(%rax)
100003e35: 75 10                       	jne	16 <_main+0x1037>
100003e37: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003e3f: c5 f8 77                    	vzeroupper
100003e42: e8 af 2f 00 00              	callq	12207 <dyld_stub_binder+0x100006df6>
100003e47: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100003e53: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100003e5b: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003e5f: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
100003e63: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100003e6b: 7e 2a                       	jle	42 <_main+0x1087>
100003e6d: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100003e75: 31 c9                       	xorl	%ecx, %ecx
100003e77: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100003e80: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003e87: 48 ff c1                    	incq	%rcx
100003e8a: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
100003e92: 48 39 d1                    	cmpq	%rdx, %rcx
100003e95: 7c e9                       	jl	-23 <_main+0x1070>
100003e97: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
100003e9f: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
100003ea7: 48 39 c7                    	cmpq	%rax, %rdi
100003eaa: 74 08                       	je	8 <_main+0x10a4>
100003eac: c5 f8 77                    	vzeroupper
100003eaf: e8 78 2f 00 00              	callq	12152 <dyld_stub_binder+0x100006e2c>
100003eb4: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100003ebc: c5 f8 77                    	vzeroupper
100003ebf: e8 ac 04 00 00              	callq	1196 <_main+0x1560>
100003ec4: 45 31 ff                    	xorl	%r15d, %r15d
100003ec7: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003ecc: 48 85 c0                    	testq	%rax, %rax
100003ecf: 0f 85 35 f9 ff ff           	jne	-1739 <_main+0x9fa>
100003ed5: e9 43 f9 ff ff              	jmp	-1725 <_main+0xa0d>
100003eda: 48 8b 3d 5f 51 00 00        	movq	20831(%rip), %rdi
100003ee1: 48 8d 35 14 50 00 00        	leaq	20500(%rip), %rsi
100003ee8: ba 0d 00 00 00              	movl	$13, %edx
100003eed: c5 f8 77                    	vzeroupper
100003ef0: e8 0b 06 00 00              	callq	1547 <_main+0x16f0>
100003ef5: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100003efd: e8 e8 2e 00 00              	callq	12008 <dyld_stub_binder+0x100006dea>
100003f02: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100003f0a: e8 01 0a 00 00              	callq	2561 <__ZN14ModelInterfaceD2Ev>
100003f0f: 48 8b 05 42 51 00 00        	movq	20802(%rip), %rax
100003f16: 48 8b 00                    	movq	(%rax), %rax
100003f19: 48 3b 84 24 e0 03 00 00     	cmpq	992(%rsp), %rax
100003f21: 75 11                       	jne	17 <_main+0x1124>
100003f23: 31 c0                       	xorl	%eax, %eax
100003f25: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100003f29: 5b                          	popq	%rbx
100003f2a: 41 5c                       	popq	%r12
100003f2c: 41 5d                       	popq	%r13
100003f2e: 41 5e                       	popq	%r14
100003f30: 41 5f                       	popq	%r15
100003f32: 5d                          	popq	%rbp
100003f33: c3                          	retq
100003f34: e8 77 2f 00 00              	callq	12151 <dyld_stub_binder+0x100006eb0>
100003f39: e9 f7 03 00 00              	jmp	1015 <_main+0x1525>
100003f3e: 48 89 c3                    	movq	%rax, %rbx
100003f41: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100003f49: 0f 84 f9 03 00 00           	je	1017 <_main+0x1538>
100003f4f: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
100003f57: e8 36 2f 00 00              	callq	12086 <dyld_stub_binder+0x100006e92>
100003f5c: e9 e7 03 00 00              	jmp	999 <_main+0x1538>
100003f61: 48 89 c3                    	movq	%rax, %rbx
100003f64: e9 df 03 00 00              	jmp	991 <_main+0x1538>
100003f69: 48 89 c7                    	movq	%rax, %rdi
100003f6c: e8 ef 03 00 00              	callq	1007 <_main+0x1550>
100003f71: 48 89 c7                    	movq	%rax, %rdi
100003f74: e8 e7 03 00 00              	callq	999 <_main+0x1550>
100003f79: 48 89 c7                    	movq	%rax, %rdi
100003f7c: e8 df 03 00 00              	callq	991 <_main+0x1550>
100003f81: 48 89 c3                    	movq	%rax, %rbx
100003f84: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003f8c: 48 85 c0                    	testq	%rax, %rax
100003f8f: 0f 85 c8 01 00 00           	jne	456 <_main+0x134d>
100003f95: e9 d6 01 00 00              	jmp	470 <_main+0x1360>
100003f9a: 48 89 c3                    	movq	%rax, %rbx
100003f9d: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
100003fa5: 48 85 c0                    	testq	%rax, %rax
100003fa8: 74 13                       	je	19 <_main+0x11ad>
100003faa: f0                          	lock
100003fab: ff 48 14                    	decl	20(%rax)
100003fae: 75 0d                       	jne	13 <_main+0x11ad>
100003fb0: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003fb8: e8 39 2e 00 00              	callq	11833 <dyld_stub_binder+0x100006df6>
100003fbd: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100003fc9: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003fcd: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100003fd5: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100003fd9: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100003fe1: 7e 21                       	jle	33 <_main+0x11f4>
100003fe3: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100003feb: 31 c9                       	xorl	%ecx, %ecx
100003fed: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003ff4: 48 ff c1                    	incq	%rcx
100003ff7: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
100003fff: 48 39 d1                    	cmpq	%rdx, %rcx
100004002: 7c e9                       	jl	-23 <_main+0x11dd>
100004004: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
10000400c: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
100004014: 48 39 c7                    	cmpq	%rax, %rdi
100004017: 0f 84 96 02 00 00           	je	662 <_main+0x14a3>
10000401d: c5 f8 77                    	vzeroupper
100004020: e8 07 2e 00 00              	callq	11783 <dyld_stub_binder+0x100006e2c>
100004025: e9 89 02 00 00              	jmp	649 <_main+0x14a3>
10000402a: 48 89 c7                    	movq	%rax, %rdi
10000402d: e8 2e 03 00 00              	callq	814 <_main+0x1550>
100004032: 48 89 c3                    	movq	%rax, %rbx
100004035: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000403a: 48 85 c0                    	testq	%rax, %rax
10000403d: 0f 85 7a 02 00 00           	jne	634 <_main+0x14ad>
100004043: e9 88 02 00 00              	jmp	648 <_main+0x14c0>
100004048: 48 89 c3                    	movq	%rax, %rbx
10000404b: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100004053: 74 1f                       	je	31 <_main+0x1264>
100004055: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
10000405d: e8 30 2e 00 00              	callq	11824 <dyld_stub_binder+0x100006e92>
100004062: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
10000406a: 75 16                       	jne	22 <_main+0x1272>
10000406c: e9 df 00 00 00              	jmp	223 <_main+0x1340>
100004071: 48 89 c3                    	movq	%rax, %rbx
100004074: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
10000407c: 0f 84 ce 00 00 00           	je	206 <_main+0x1340>
100004082: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
10000408a: e9 aa 00 00 00              	jmp	170 <_main+0x1329>
10000408f: 48 89 c3                    	movq	%rax, %rbx
100004092: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
10000409a: 75 23                       	jne	35 <_main+0x12af>
10000409c: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000040a4: 75 3f                       	jne	63 <_main+0x12d5>
1000040a6: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
1000040ae: 75 5b                       	jne	91 <_main+0x12fb>
1000040b0: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
1000040b8: 75 77                       	jne	119 <_main+0x1321>
1000040ba: e9 91 00 00 00              	jmp	145 <_main+0x1340>
1000040bf: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
1000040c7: e8 c6 2d 00 00              	callq	11718 <dyld_stub_binder+0x100006e92>
1000040cc: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000040d4: 74 d0                       	je	-48 <_main+0x1296>
1000040d6: eb 0d                       	jmp	13 <_main+0x12d5>
1000040d8: 48 89 c3                    	movq	%rax, %rbx
1000040db: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000040e3: 74 c1                       	je	-63 <_main+0x1296>
1000040e5: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
1000040ed: e8 a0 2d 00 00              	callq	11680 <dyld_stub_binder+0x100006e92>
1000040f2: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
1000040fa: 74 b4                       	je	-76 <_main+0x12a0>
1000040fc: eb 0d                       	jmp	13 <_main+0x12fb>
1000040fe: 48 89 c3                    	movq	%rax, %rbx
100004101: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100004109: 74 a5                       	je	-91 <_main+0x12a0>
10000410b: 48 8b bc 24 50 01 00 00     	movq	336(%rsp), %rdi
100004113: e8 7a 2d 00 00              	callq	11642 <dyld_stub_binder+0x100006e92>
100004118: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100004120: 75 0f                       	jne	15 <_main+0x1321>
100004122: eb 2c                       	jmp	44 <_main+0x1340>
100004124: 48 89 c3                    	movq	%rax, %rbx
100004127: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
10000412f: 74 1f                       	je	31 <_main+0x1340>
100004131: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
100004139: e8 54 2d 00 00              	callq	11604 <dyld_stub_binder+0x100006e92>
10000413e: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100004146: 48 85 c0                    	testq	%rax, %rax
100004149: 75 12                       	jne	18 <_main+0x134d>
10000414b: eb 23                       	jmp	35 <_main+0x1360>
10000414d: 48 89 c3                    	movq	%rax, %rbx
100004150: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100004158: 48 85 c0                    	testq	%rax, %rax
10000415b: 74 13                       	je	19 <_main+0x1360>
10000415d: f0                          	lock
10000415e: ff 48 14                    	decl	20(%rax)
100004161: 75 0d                       	jne	13 <_main+0x1360>
100004163: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
10000416b: e8 86 2c 00 00              	callq	11398 <dyld_stub_binder+0x100006df6>
100004170: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
10000417c: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100004180: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
100004188: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
10000418d: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
100004195: 7e 21                       	jle	33 <_main+0x13a8>
100004197: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
10000419f: 31 c9                       	xorl	%ecx, %ecx
1000041a1: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000041a8: 48 ff c1                    	incq	%rcx
1000041ab: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
1000041b3: 48 39 d1                    	cmpq	%rdx, %rcx
1000041b6: 7c e9                       	jl	-23 <_main+0x1391>
1000041b8: 48 8b bc 24 c8 00 00 00     	movq	200(%rsp), %rdi
1000041c0: 48 8d 84 24 d0 00 00 00     	leaq	208(%rsp), %rax
1000041c8: 48 39 c7                    	cmpq	%rax, %rdi
1000041cb: 74 21                       	je	33 <_main+0x13de>
1000041cd: c5 f8 77                    	vzeroupper
1000041d0: e8 57 2c 00 00              	callq	11351 <dyld_stub_binder+0x100006e2c>
1000041d5: eb 17                       	jmp	23 <_main+0x13de>
1000041d7: 48 89 c7                    	movq	%rax, %rdi
1000041da: e8 81 01 00 00              	callq	385 <_main+0x1550>
1000041df: eb 0a                       	jmp	10 <_main+0x13db>
1000041e1: eb 08                       	jmp	8 <_main+0x13db>
1000041e3: 48 89 c3                    	movq	%rax, %rbx
1000041e6: e9 8a 00 00 00              	jmp	138 <_main+0x1465>
1000041eb: 48 89 c3                    	movq	%rax, %rbx
1000041ee: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
1000041f6: 48 85 c0                    	testq	%rax, %rax
1000041f9: 74 16                       	je	22 <_main+0x1401>
1000041fb: f0                          	lock
1000041fc: ff 48 14                    	decl	20(%rax)
1000041ff: 75 10                       	jne	16 <_main+0x1401>
100004201: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100004209: c5 f8 77                    	vzeroupper
10000420c: e8 e5 2b 00 00              	callq	11237 <dyld_stub_binder+0x100006df6>
100004211: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
10000421d: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100004221: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100004229: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
10000422d: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100004235: 7e 21                       	jle	33 <_main+0x1448>
100004237: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
10000423f: 31 c9                       	xorl	%ecx, %ecx
100004241: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004248: 48 ff c1                    	incq	%rcx
10000424b: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
100004253: 48 39 d1                    	cmpq	%rdx, %rcx
100004256: 7c e9                       	jl	-23 <_main+0x1431>
100004258: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
100004260: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
100004268: 48 39 c7                    	cmpq	%rax, %rdi
10000426b: 74 08                       	je	8 <_main+0x1465>
10000426d: c5 f8 77                    	vzeroupper
100004270: e8 b7 2b 00 00              	callq	11191 <dyld_stub_binder+0x100006e2c>
100004275: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
10000427d: c5 f8 77                    	vzeroupper
100004280: e8 eb 00 00 00              	callq	235 <_main+0x1560>
100004285: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000428a: 48 85 c0                    	testq	%rax, %rax
10000428d: 75 2e                       	jne	46 <_main+0x14ad>
10000428f: eb 3f                       	jmp	63 <_main+0x14c0>
100004291: 48 89 c7                    	movq	%rax, %rdi
100004294: e8 c7 00 00 00              	callq	199 <_main+0x1550>
100004299: 48 89 c3                    	movq	%rax, %rbx
10000429c: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000042a1: 48 85 c0                    	testq	%rax, %rax
1000042a4: 75 17                       	jne	23 <_main+0x14ad>
1000042a6: eb 28                       	jmp	40 <_main+0x14c0>
1000042a8: 48 89 c7                    	movq	%rax, %rdi
1000042ab: e8 b0 00 00 00              	callq	176 <_main+0x1550>
1000042b0: 48 89 c3                    	movq	%rax, %rbx
1000042b3: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000042b8: 48 85 c0                    	testq	%rax, %rax
1000042bb: 74 13                       	je	19 <_main+0x14c0>
1000042bd: f0                          	lock
1000042be: ff 48 14                    	decl	20(%rax)
1000042c1: 75 0d                       	jne	13 <_main+0x14c0>
1000042c3: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
1000042c8: c5 f8 77                    	vzeroupper
1000042cb: e8 26 2b 00 00              	callq	11046 <dyld_stub_binder+0x100006df6>
1000042d0: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
1000042d9: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000042dd: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
1000042e2: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
1000042e7: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
1000042ec: 7e 26                       	jle	38 <_main+0x1504>
1000042ee: 48 8b 44 24 58              	movq	88(%rsp), %rax
1000042f3: 31 c9                       	xorl	%ecx, %ecx
1000042f5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000042ff: 90                          	nop
100004300: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004307: 48 ff c1                    	incq	%rcx
10000430a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000430f: 48 39 d1                    	cmpq	%rdx, %rcx
100004312: 7c ec                       	jl	-20 <_main+0x14f0>
100004314: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100004319: 48 8d 44 24 68              	leaq	104(%rsp), %rax
10000431e: 48 39 c7                    	cmpq	%rax, %rdi
100004321: 74 15                       	je	21 <_main+0x1528>
100004323: c5 f8 77                    	vzeroupper
100004326: e8 01 2b 00 00              	callq	11009 <dyld_stub_binder+0x100006e2c>
10000432b: eb 0b                       	jmp	11 <_main+0x1528>
10000432d: 48 89 c7                    	movq	%rax, %rdi
100004330: e8 2b 00 00 00              	callq	43 <_main+0x1550>
100004335: 48 89 c3                    	movq	%rax, %rbx
100004338: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100004340: c5 f8 77                    	vzeroupper
100004343: e8 a2 2a 00 00              	callq	10914 <dyld_stub_binder+0x100006dea>
100004348: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100004350: e8 bb 05 00 00              	callq	1467 <__ZN14ModelInterfaceD2Ev>
100004355: 48 89 df                    	movq	%rbx, %rdi
100004358: e8 7b 2a 00 00              	callq	10875 <dyld_stub_binder+0x100006dd8>
10000435d: 0f 0b                       	ud2
10000435f: 90                          	nop
100004360: 50                          	pushq	%rax
100004361: e8 3e 2b 00 00              	callq	11070 <dyld_stub_binder+0x100006ea4>
100004366: e8 21 2b 00 00              	callq	11041 <dyld_stub_binder+0x100006e8c>
10000436b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004370: 55                          	pushq	%rbp
100004371: 48 89 e5                    	movq	%rsp, %rbp
100004374: 53                          	pushq	%rbx
100004375: 50                          	pushq	%rax
100004376: 48 89 fb                    	movq	%rdi, %rbx
100004379: 48 8b 87 08 01 00 00        	movq	264(%rdi), %rax
100004380: 48 85 c0                    	testq	%rax, %rax
100004383: 74 12                       	je	18 <_main+0x1587>
100004385: f0                          	lock
100004386: ff 48 14                    	decl	20(%rax)
100004389: 75 0c                       	jne	12 <_main+0x1587>
10000438b: 48 8d bb d0 00 00 00        	leaq	208(%rbx), %rdi
100004392: e8 5f 2a 00 00              	callq	10847 <dyld_stub_binder+0x100006df6>
100004397: 48 c7 83 08 01 00 00 00 00 00 00    	movq	$0, 264(%rbx)
1000043a2: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000043a6: c5 fc 11 83 e0 00 00 00     	vmovups	%ymm0, 224(%rbx)
1000043ae: 83 bb d4 00 00 00 00        	cmpl	$0, 212(%rbx)
1000043b5: 7e 1f                       	jle	31 <_main+0x15c6>
1000043b7: 48 8b 83 10 01 00 00        	movq	272(%rbx), %rax
1000043be: 31 c9                       	xorl	%ecx, %ecx
1000043c0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000043c7: 48 ff c1                    	incq	%rcx
1000043ca: 48 63 93 d4 00 00 00        	movslq	212(%rbx), %rdx
1000043d1: 48 39 d1                    	cmpq	%rdx, %rcx
1000043d4: 7c ea                       	jl	-22 <_main+0x15b0>
1000043d6: 48 8b bb 18 01 00 00        	movq	280(%rbx), %rdi
1000043dd: 48 8d 83 20 01 00 00        	leaq	288(%rbx), %rax
1000043e4: 48 39 c7                    	cmpq	%rax, %rdi
1000043e7: 74 08                       	je	8 <_main+0x15e1>
1000043e9: c5 f8 77                    	vzeroupper
1000043ec: e8 3b 2a 00 00              	callq	10811 <dyld_stub_binder+0x100006e2c>
1000043f1: 48 8b 83 a8 00 00 00        	movq	168(%rbx), %rax
1000043f8: 48 85 c0                    	testq	%rax, %rax
1000043fb: 74 12                       	je	18 <_main+0x15ff>
1000043fd: f0                          	lock
1000043fe: ff 48 14                    	decl	20(%rax)
100004401: 75 0c                       	jne	12 <_main+0x15ff>
100004403: 48 8d 7b 70                 	leaq	112(%rbx), %rdi
100004407: c5 f8 77                    	vzeroupper
10000440a: e8 e7 29 00 00              	callq	10727 <dyld_stub_binder+0x100006df6>
10000440f: 48 c7 83 a8 00 00 00 00 00 00 00    	movq	$0, 168(%rbx)
10000441a: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000441e: c5 fc 11 83 80 00 00 00     	vmovups	%ymm0, 128(%rbx)
100004426: 83 7b 74 00                 	cmpl	$0, 116(%rbx)
10000442a: 7e 27                       	jle	39 <_main+0x1643>
10000442c: 48 8b 83 b0 00 00 00        	movq	176(%rbx), %rax
100004433: 31 c9                       	xorl	%ecx, %ecx
100004435: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000443f: 90                          	nop
100004440: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004447: 48 ff c1                    	incq	%rcx
10000444a: 48 63 53 74                 	movslq	116(%rbx), %rdx
10000444e: 48 39 d1                    	cmpq	%rdx, %rcx
100004451: 7c ed                       	jl	-19 <_main+0x1630>
100004453: 48 8b bb b8 00 00 00        	movq	184(%rbx), %rdi
10000445a: 48 8d 83 c0 00 00 00        	leaq	192(%rbx), %rax
100004461: 48 39 c7                    	cmpq	%rax, %rdi
100004464: 74 08                       	je	8 <_main+0x165e>
100004466: c5 f8 77                    	vzeroupper
100004469: e8 be 29 00 00              	callq	10686 <dyld_stub_binder+0x100006e2c>
10000446e: 48 8b 43 48                 	movq	72(%rbx), %rax
100004472: 48 85 c0                    	testq	%rax, %rax
100004475: 74 12                       	je	18 <_main+0x1679>
100004477: f0                          	lock
100004478: ff 48 14                    	decl	20(%rax)
10000447b: 75 0c                       	jne	12 <_main+0x1679>
10000447d: 48 8d 7b 10                 	leaq	16(%rbx), %rdi
100004481: c5 f8 77                    	vzeroupper
100004484: e8 6d 29 00 00              	callq	10605 <dyld_stub_binder+0x100006df6>
100004489: 48 c7 43 48 00 00 00 00     	movq	$0, 72(%rbx)
100004491: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004495: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
10000449a: 83 7b 14 00                 	cmpl	$0, 20(%rbx)
10000449e: 7e 23                       	jle	35 <_main+0x16b3>
1000044a0: 48 8b 43 50                 	movq	80(%rbx), %rax
1000044a4: 31 c9                       	xorl	%ecx, %ecx
1000044a6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000044b0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000044b7: 48 ff c1                    	incq	%rcx
1000044ba: 48 63 53 14                 	movslq	20(%rbx), %rdx
1000044be: 48 39 d1                    	cmpq	%rdx, %rcx
1000044c1: 7c ed                       	jl	-19 <_main+0x16a0>
1000044c3: 48 8b 7b 58                 	movq	88(%rbx), %rdi
1000044c7: 48 83 c3 60                 	addq	$96, %rbx
1000044cb: 48 39 df                    	cmpq	%rbx, %rdi
1000044ce: 74 08                       	je	8 <_main+0x16c8>
1000044d0: c5 f8 77                    	vzeroupper
1000044d3: e8 54 29 00 00              	callq	10580 <dyld_stub_binder+0x100006e2c>
1000044d8: 48 83 c4 08                 	addq	$8, %rsp
1000044dc: 5b                          	popq	%rbx
1000044dd: 5d                          	popq	%rbp
1000044de: c5 f8 77                    	vzeroupper
1000044e1: c3                          	retq
1000044e2: 48 89 c7                    	movq	%rax, %rdi
1000044e5: e8 76 fe ff ff              	callq	-394 <_main+0x1550>
1000044ea: 48 89 c7                    	movq	%rax, %rdi
1000044ed: e8 6e fe ff ff              	callq	-402 <_main+0x1550>
1000044f2: 48 89 c7                    	movq	%rax, %rdi
1000044f5: e8 66 fe ff ff              	callq	-410 <_main+0x1550>
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
100004521: e8 36 29 00 00              	callq	10550 <dyld_stub_binder+0x100006e5c>
100004526: 80 7d b0 00                 	cmpb	$0, -80(%rbp)
10000452a: 0f 84 ae 00 00 00           	je	174 <_main+0x17ce>
100004530: 48 8b 03                    	movq	(%rbx), %rax
100004533: 48 8b 40 e8                 	movq	-24(%rax), %rax
100004537: 4c 8d 24 03                 	leaq	(%rbx,%rax), %r12
10000453b: 48 8b 7c 03 28              	movq	40(%rbx,%rax), %rdi
100004540: 44 8b 6c 03 08              	movl	8(%rbx,%rax), %r13d
100004545: 8b 84 03 90 00 00 00        	movl	144(%rbx,%rax), %eax
10000454c: 83 f8 ff                    	cmpl	$-1, %eax
10000454f: 75 4a                       	jne	74 <_main+0x178b>
100004551: 48 89 7d c0                 	movq	%rdi, -64(%rbp)
100004555: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004559: 4c 89 e6                    	movq	%r12, %rsi
10000455c: e8 e3 28 00 00              	callq	10467 <dyld_stub_binder+0x100006e44>
100004561: 48 8b 35 e0 4a 00 00        	movq	19168(%rip), %rsi
100004568: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
10000456c: e8 cd 28 00 00              	callq	10445 <dyld_stub_binder+0x100006e3e>
100004571: 48 8b 08                    	movq	(%rax), %rcx
100004574: 48 89 c7                    	movq	%rax, %rdi
100004577: be 20 00 00 00              	movl	$32, %esi
10000457c: ff 51 38                    	callq	*56(%rcx)
10000457f: 88 45 d7                    	movb	%al, -41(%rbp)
100004582: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004586: e8 e3 28 00 00              	callq	10467 <dyld_stub_binder+0x100006e6e>
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
1000045bd: e8 9e 00 00 00              	callq	158 <_main+0x1850>
1000045c2: 48 85 c0                    	testq	%rax, %rax
1000045c5: 75 17                       	jne	23 <_main+0x17ce>
1000045c7: 48 8b 03                    	movq	(%rbx), %rax
1000045ca: 48 8b 40 e8                 	movq	-24(%rax), %rax
1000045ce: 48 8d 3c 03                 	leaq	(%rbx,%rax), %rdi
1000045d2: 8b 74 03 20                 	movl	32(%rbx,%rax), %esi
1000045d6: 83 ce 05                    	orl	$5, %esi
1000045d9: e8 9c 28 00 00              	callq	10396 <dyld_stub_binder+0x100006e7a>
1000045de: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
1000045e2: e8 7b 28 00 00              	callq	10363 <dyld_stub_binder+0x100006e62>
1000045e7: 48 89 d8                    	movq	%rbx, %rax
1000045ea: 48 83 c4 28                 	addq	$40, %rsp
1000045ee: 5b                          	popq	%rbx
1000045ef: 41 5c                       	popq	%r12
1000045f1: 41 5d                       	popq	%r13
1000045f3: 41 5e                       	popq	%r14
1000045f5: 41 5f                       	popq	%r15
1000045f7: 5d                          	popq	%rbp
1000045f8: c3                          	retq
1000045f9: eb 0e                       	jmp	14 <_main+0x17f9>
1000045fb: 49 89 c6                    	movq	%rax, %r14
1000045fe: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004602: e8 67 28 00 00              	callq	10343 <dyld_stub_binder+0x100006e6e>
100004607: eb 03                       	jmp	3 <_main+0x17fc>
100004609: 49 89 c6                    	movq	%rax, %r14
10000460c: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100004610: e8 4d 28 00 00              	callq	10317 <dyld_stub_binder+0x100006e62>
100004615: eb 03                       	jmp	3 <_main+0x180a>
100004617: 49 89 c6                    	movq	%rax, %r14
10000461a: 4c 89 f7                    	movq	%r14, %rdi
10000461d: e8 82 28 00 00              	callq	10370 <dyld_stub_binder+0x100006ea4>
100004622: 48 8b 03                    	movq	(%rbx), %rax
100004625: 48 8b 78 e8                 	movq	-24(%rax), %rdi
100004629: 48 01 df                    	addq	%rbx, %rdi
10000462c: e8 43 28 00 00              	callq	10307 <dyld_stub_binder+0x100006e74>
100004631: e8 74 28 00 00              	callq	10356 <dyld_stub_binder+0x100006eaa>
100004636: eb af                       	jmp	-81 <_main+0x17d7>
100004638: 48 89 c3                    	movq	%rax, %rbx
10000463b: e8 6a 28 00 00              	callq	10346 <dyld_stub_binder+0x100006eaa>
100004640: 48 89 df                    	movq	%rbx, %rdi
100004643: e8 90 27 00 00              	callq	10128 <dyld_stub_binder+0x100006dd8>
100004648: 0f 0b                       	ud2
10000464a: 48 89 c7                    	movq	%rax, %rdi
10000464d: e8 0e fd ff ff              	callq	-754 <_main+0x1550>
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
100004674: 0f 84 17 01 00 00           	je	279 <_main+0x1981>
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
1000046a8: 7e 15                       	jle	21 <_main+0x18af>
1000046aa: 49 8b 06                    	movq	(%r14), %rax
1000046ad: 4c 89 f7                    	movq	%r14, %rdi
1000046b0: 48 89 da                    	movq	%rbx, %rdx
1000046b3: ff 50 60                    	callq	*96(%rax)
1000046b6: 48 39 d8                    	cmpq	%rbx, %rax
1000046b9: 0f 85 d2 00 00 00           	jne	210 <_main+0x1981>
1000046bf: 4d 85 ed                    	testq	%r13, %r13
1000046c2: 0f 8e a1 00 00 00           	jle	161 <_main+0x1959>
1000046c8: 4c 89 65 b0                 	movq	%r12, -80(%rbp)
1000046cc: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000046d0: c5 f8 29 45 c0              	vmovaps	%xmm0, -64(%rbp)
1000046d5: 48 c7 45 d0 00 00 00 00     	movq	$0, -48(%rbp)
1000046dd: 49 83 fd 17                 	cmpq	$23, %r13
1000046e1: 73 12                       	jae	18 <_main+0x18e5>
1000046e3: 43 8d 44 2d 00              	leal	(%r13,%r13), %eax
1000046e8: 88 45 c0                    	movb	%al, -64(%rbp)
1000046eb: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
1000046ef: 4c 8d 65 c1                 	leaq	-63(%rbp), %r12
1000046f3: eb 27                       	jmp	39 <_main+0x190c>
1000046f5: 49 8d 5d 10                 	leaq	16(%r13), %rbx
1000046f9: 48 83 e3 f0                 	andq	$-16, %rbx
1000046fd: 48 89 df                    	movq	%rbx, %rdi
100004700: e8 99 27 00 00              	callq	10137 <dyld_stub_binder+0x100006e9e>
100004705: 49 89 c4                    	movq	%rax, %r12
100004708: 48 89 45 d0                 	movq	%rax, -48(%rbp)
10000470c: 48 83 cb 01                 	orq	$1, %rbx
100004710: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
100004714: 4c 89 6d c8                 	movq	%r13, -56(%rbp)
100004718: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
10000471c: 0f b6 75 bc                 	movzbl	-68(%rbp), %esi
100004720: 4c 89 e7                    	movq	%r12, %rdi
100004723: 4c 89 ea                    	movq	%r13, %rdx
100004726: e8 8b 27 00 00              	callq	10123 <dyld_stub_binder+0x100006eb6>
10000472b: 43 c6 04 2c 00              	movb	$0, (%r12,%r13)
100004730: f6 45 c0 01                 	testb	$1, -64(%rbp)
100004734: 74 06                       	je	6 <_main+0x192c>
100004736: 48 8b 5d d0                 	movq	-48(%rbp), %rbx
10000473a: eb 03                       	jmp	3 <_main+0x192f>
10000473c: 48 ff c3                    	incq	%rbx
10000473f: 4c 8b 65 b0                 	movq	-80(%rbp), %r12
100004743: 49 8b 06                    	movq	(%r14), %rax
100004746: 4c 89 f7                    	movq	%r14, %rdi
100004749: 48 89 de                    	movq	%rbx, %rsi
10000474c: 4c 89 ea                    	movq	%r13, %rdx
10000474f: ff 50 60                    	callq	*96(%rax)
100004752: 48 89 c3                    	movq	%rax, %rbx
100004755: f6 45 c0 01                 	testb	$1, -64(%rbp)
100004759: 74 09                       	je	9 <_main+0x1954>
10000475b: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
10000475f: e8 2e 27 00 00              	callq	10030 <dyld_stub_binder+0x100006e92>
100004764: 4c 39 eb                    	cmpq	%r13, %rbx
100004767: 75 28                       	jne	40 <_main+0x1981>
100004769: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
10000476d: 49 29 f7                    	subq	%rsi, %r15
100004770: 4d 85 ff                    	testq	%r15, %r15
100004773: 7e 11                       	jle	17 <_main+0x1976>
100004775: 49 8b 06                    	movq	(%r14), %rax
100004778: 4c 89 f7                    	movq	%r14, %rdi
10000477b: 4c 89 fa                    	movq	%r15, %rdx
10000477e: ff 50 60                    	callq	*96(%rax)
100004781: 4c 39 f8                    	cmpq	%r15, %rax
100004784: 75 0b                       	jne	11 <_main+0x1981>
100004786: 49 c7 44 24 18 00 00 00 00  	movq	$0, 24(%r12)
10000478f: eb 03                       	jmp	3 <_main+0x1984>
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
1000047ad: 74 09                       	je	9 <_main+0x19a8>
1000047af: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
1000047b3: e8 da 26 00 00              	callq	9946 <dyld_stub_binder+0x100006e92>
1000047b8: 48 89 df                    	movq	%rbx, %rdi
1000047bb: e8 18 26 00 00              	callq	9752 <dyld_stub_binder+0x100006dd8>
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
1000047de: 74 02                       	je	2 <_main+0x19d2>
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
1000047fe: 74 02                       	je	2 <_main+0x19f2>
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
10000481e: 74 02                       	je	2 <_main+0x1a12>
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
10000483e: 74 02                       	je	2 <_main+0x1a32>
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
10000485e: 74 02                       	je	2 <_main+0x1a52>
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
10000487e: 74 02                       	je	2 <_main+0x1a72>
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
10000489e: 74 02                       	je	2 <_main+0x1a92>
1000048a0: 5d                          	popq	%rbp
1000048a1: c3                          	retq
1000048a2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048a9: 5d                          	popq	%rbp
1000048aa: c3                          	retq
1000048ab: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048b0: 55                          	pushq	%rbp
1000048b1: 48 89 e5                    	movq	%rsp, %rbp
1000048b4: 48 8b 05 5d 47 00 00        	movq	18269(%rip), %rax
1000048bb: 80 38 00                    	cmpb	$0, (%rax)
1000048be: 74 02                       	je	2 <_main+0x1ab2>
1000048c0: 5d                          	popq	%rbp
1000048c1: c3                          	retq
1000048c2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048c9: 5d                          	popq	%rbp
1000048ca: c3                          	retq
1000048cb: 90                          	nop
1000048cc: 90                          	nop
1000048cd: 90                          	nop
1000048ce: 90                          	nop
1000048cf: 90                          	nop

00000001000048d0 __ZN14ModelInterfaceC2Ev:
1000048d0: 55                          	pushq	%rbp
1000048d1: 48 89 e5                    	movq	%rsp, %rbp
1000048d4: 48 8d 05 dd 47 00 00        	leaq	18397(%rip), %rax
1000048db: 48 89 07                    	movq	%rax, (%rdi)
1000048de: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000048e2: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
1000048e7: 5d                          	popq	%rbp
1000048e8: c3                          	retq
1000048e9: 0f 1f 80 00 00 00 00        	nopl	(%rax)

00000001000048f0 __ZN14ModelInterfaceC1Ev:
1000048f0: 55                          	pushq	%rbp
1000048f1: 48 89 e5                    	movq	%rsp, %rbp
1000048f4: 48 8d 05 bd 47 00 00        	leaq	18365(%rip), %rax
1000048fb: 48 89 07                    	movq	%rax, (%rdi)
1000048fe: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004902: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
100004907: 5d                          	popq	%rbp
100004908: c3                          	retq
100004909: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004910 __ZN14ModelInterfaceD2Ev:
100004910: 55                          	pushq	%rbp
100004911: 48 89 e5                    	movq	%rsp, %rbp
100004914: 53                          	pushq	%rbx
100004915: 50                          	pushq	%rax
100004916: 48 89 fb                    	movq	%rdi, %rbx
100004919: 48 8d 05 98 47 00 00        	leaq	18328(%rip), %rax
100004920: 48 89 07                    	movq	%rax, (%rdi)
100004923: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100004927: 48 85 ff                    	testq	%rdi, %rdi
10000492a: 74 05                       	je	5 <__ZN14ModelInterfaceD2Ev+0x21>
10000492c: e8 61 25 00 00              	callq	9569 <dyld_stub_binder+0x100006e92>
100004931: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100004935: 48 83 c4 08                 	addq	$8, %rsp
100004939: 48 85 ff                    	testq	%rdi, %rdi
10000493c: 74 07                       	je	7 <__ZN14ModelInterfaceD2Ev+0x35>
10000493e: 5b                          	popq	%rbx
10000493f: 5d                          	popq	%rbp
100004940: e9 4d 25 00 00              	jmp	9549 <dyld_stub_binder+0x100006e92>
100004945: 5b                          	popq	%rbx
100004946: 5d                          	popq	%rbp
100004947: c3                          	retq
100004948: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100004950 __ZN14ModelInterfaceD1Ev:
100004950: 55                          	pushq	%rbp
100004951: 48 89 e5                    	movq	%rsp, %rbp
100004954: 53                          	pushq	%rbx
100004955: 50                          	pushq	%rax
100004956: 48 89 fb                    	movq	%rdi, %rbx
100004959: 48 8d 05 58 47 00 00        	leaq	18264(%rip), %rax
100004960: 48 89 07                    	movq	%rax, (%rdi)
100004963: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100004967: 48 85 ff                    	testq	%rdi, %rdi
10000496a: 74 05                       	je	5 <__ZN14ModelInterfaceD1Ev+0x21>
10000496c: e8 21 25 00 00              	callq	9505 <dyld_stub_binder+0x100006e92>
100004971: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100004975: 48 83 c4 08                 	addq	$8, %rsp
100004979: 48 85 ff                    	testq	%rdi, %rdi
10000497c: 74 07                       	je	7 <__ZN14ModelInterfaceD1Ev+0x35>
10000497e: 5b                          	popq	%rbx
10000497f: 5d                          	popq	%rbp
100004980: e9 0d 25 00 00              	jmp	9485 <dyld_stub_binder+0x100006e92>
100004985: 5b                          	popq	%rbx
100004986: 5d                          	popq	%rbp
100004987: c3                          	retq
100004988: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100004990 __ZN14ModelInterfaceD0Ev:
100004990: 55                          	pushq	%rbp
100004991: 48 89 e5                    	movq	%rsp, %rbp
100004994: 53                          	pushq	%rbx
100004995: 50                          	pushq	%rax
100004996: 48 89 fb                    	movq	%rdi, %rbx
100004999: 48 8d 05 18 47 00 00        	leaq	18200(%rip), %rax
1000049a0: 48 89 07                    	movq	%rax, (%rdi)
1000049a3: 48 8b 7f 28                 	movq	40(%rdi), %rdi
1000049a7: 48 85 ff                    	testq	%rdi, %rdi
1000049aa: 74 05                       	je	5 <__ZN14ModelInterfaceD0Ev+0x21>
1000049ac: e8 e1 24 00 00              	callq	9441 <dyld_stub_binder+0x100006e92>
1000049b1: 48 8b 7b 30                 	movq	48(%rbx), %rdi
1000049b5: 48 85 ff                    	testq	%rdi, %rdi
1000049b8: 74 05                       	je	5 <__ZN14ModelInterfaceD0Ev+0x2f>
1000049ba: e8 d3 24 00 00              	callq	9427 <dyld_stub_binder+0x100006e92>
1000049bf: 48 89 df                    	movq	%rbx, %rdi
1000049c2: 48 83 c4 08                 	addq	$8, %rsp
1000049c6: 5b                          	popq	%rbx
1000049c7: 5d                          	popq	%rbp
1000049c8: e9 c5 24 00 00              	jmp	9413 <dyld_stub_binder+0x100006e92>
1000049cd: 0f 1f 00                    	nopl	(%rax)

00000001000049d0 __ZN14ModelInterface7forwardEv:
1000049d0: 55                          	pushq	%rbp
1000049d1: 48 89 e5                    	movq	%rsp, %rbp
1000049d4: 5d                          	popq	%rbp
1000049d5: c3                          	retq
1000049d6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

00000001000049e0 __ZN14ModelInterface12input_bufferEv:
1000049e0: 55                          	pushq	%rbp
1000049e1: 48 89 e5                    	movq	%rsp, %rbp
1000049e4: 0f b6 47 24                 	movzbl	36(%rdi), %eax
1000049e8: 48 8b 44 c7 28              	movq	40(%rdi,%rax,8), %rax
1000049ed: 5d                          	popq	%rbp
1000049ee: c3                          	retq
1000049ef: 90                          	nop

00000001000049f0 __ZN14ModelInterface13output_bufferEv:
1000049f0: 55                          	pushq	%rbp
1000049f1: 48 89 e5                    	movq	%rsp, %rbp
1000049f4: 31 c0                       	xorl	%eax, %eax
1000049f6: 80 7f 24 00                 	cmpb	$0, 36(%rdi)
1000049fa: 0f 94 c0                    	sete	%al
1000049fd: 48 8b 44 c7 28              	movq	40(%rdi,%rax,8), %rax
100004a02: 5d                          	popq	%rbp
100004a03: c3                          	retq
100004a04: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004a0e: 66 90                       	nop

0000000100004a10 __ZN14ModelInterface11init_bufferEj:
100004a10: 55                          	pushq	%rbp
100004a11: 48 89 e5                    	movq	%rsp, %rbp
100004a14: 41 57                       	pushq	%r15
100004a16: 41 56                       	pushq	%r14
100004a18: 41 54                       	pushq	%r12
100004a1a: 53                          	pushq	%rbx
100004a1b: 41 89 f7                    	movl	%esi, %r15d
100004a1e: 48 89 fb                    	movq	%rdi, %rbx
100004a21: c6 47 24 00                 	movb	$0, 36(%rdi)
100004a25: 41 89 f6                    	movl	%esi, %r14d
100004a28: 4c 89 f7                    	movq	%r14, %rdi
100004a2b: e8 68 24 00 00              	callq	9320 <dyld_stub_binder+0x100006e98>
100004a30: 49 89 c4                    	movq	%rax, %r12
100004a33: 48 89 43 28                 	movq	%rax, 40(%rbx)
100004a37: 4c 89 f7                    	movq	%r14, %rdi
100004a3a: e8 59 24 00 00              	callq	9305 <dyld_stub_binder+0x100006e98>
100004a3f: 48 89 43 30                 	movq	%rax, 48(%rbx)
100004a43: 45 85 ff                    	testl	%r15d, %r15d
100004a46: 0f 84 44 01 00 00           	je	324 <__ZN14ModelInterface11init_bufferEj+0x180>
100004a4c: 41 c6 04 24 00              	movb	$0, (%r12)
100004a51: 41 83 ff 01                 	cmpl	$1, %r15d
100004a55: 0f 84 95 00 00 00           	je	149 <__ZN14ModelInterface11init_bufferEj+0xe0>
100004a5b: 41 8d 46 ff                 	leal	-1(%r14), %eax
100004a5f: 49 8d 56 fe                 	leaq	-2(%r14), %rdx
100004a63: 83 e0 07                    	andl	$7, %eax
100004a66: b9 01 00 00 00              	movl	$1, %ecx
100004a6b: 48 83 fa 07                 	cmpq	$7, %rdx
100004a6f: 72 63                       	jb	99 <__ZN14ModelInterface11init_bufferEj+0xc4>
100004a71: 48 89 c2                    	movq	%rax, %rdx
100004a74: 48 f7 d2                    	notq	%rdx
100004a77: 4c 01 f2                    	addq	%r14, %rdx
100004a7a: 31 c9                       	xorl	%ecx, %ecx
100004a7c: 0f 1f 40 00                 	nopl	(%rax)
100004a80: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004a84: c6 44 0e 01 00              	movb	$0, 1(%rsi,%rcx)
100004a89: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004a8d: c6 44 0e 02 00              	movb	$0, 2(%rsi,%rcx)
100004a92: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004a96: c6 44 0e 03 00              	movb	$0, 3(%rsi,%rcx)
100004a9b: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004a9f: c6 44 0e 04 00              	movb	$0, 4(%rsi,%rcx)
100004aa4: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004aa8: c6 44 0e 05 00              	movb	$0, 5(%rsi,%rcx)
100004aad: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004ab1: c6 44 0e 06 00              	movb	$0, 6(%rsi,%rcx)
100004ab6: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004aba: c6 44 0e 07 00              	movb	$0, 7(%rsi,%rcx)
100004abf: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004ac3: c6 44 0e 08 00              	movb	$0, 8(%rsi,%rcx)
100004ac8: 48 83 c1 08                 	addq	$8, %rcx
100004acc: 48 39 ca                    	cmpq	%rcx, %rdx
100004acf: 75 af                       	jne	-81 <__ZN14ModelInterface11init_bufferEj+0x70>
100004ad1: 48 ff c1                    	incq	%rcx
100004ad4: 48 85 c0                    	testq	%rax, %rax
100004ad7: 74 17                       	je	23 <__ZN14ModelInterface11init_bufferEj+0xe0>
100004ad9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004ae0: 48 8b 53 28                 	movq	40(%rbx), %rdx
100004ae4: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004ae8: 48 ff c1                    	incq	%rcx
100004aeb: 48 ff c8                    	decq	%rax
100004aee: 75 f0                       	jne	-16 <__ZN14ModelInterface11init_bufferEj+0xd0>
100004af0: 45 85 ff                    	testl	%r15d, %r15d
100004af3: 0f 84 97 00 00 00           	je	151 <__ZN14ModelInterface11init_bufferEj+0x180>
100004af9: 49 8d 4e ff                 	leaq	-1(%r14), %rcx
100004afd: 44 89 f0                    	movl	%r14d, %eax
100004b00: 83 e0 07                    	andl	$7, %eax
100004b03: 48 83 f9 07                 	cmpq	$7, %rcx
100004b07: 73 0c                       	jae	12 <__ZN14ModelInterface11init_bufferEj+0x105>
100004b09: 31 c9                       	xorl	%ecx, %ecx
100004b0b: 48 85 c0                    	testq	%rax, %rax
100004b0e: 75 70                       	jne	112 <__ZN14ModelInterface11init_bufferEj+0x170>
100004b10: e9 7b 00 00 00              	jmp	123 <__ZN14ModelInterface11init_bufferEj+0x180>
100004b15: 49 29 c6                    	subq	%rax, %r14
100004b18: 31 c9                       	xorl	%ecx, %ecx
100004b1a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100004b20: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b24: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004b28: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b2c: c6 44 0a 01 00              	movb	$0, 1(%rdx,%rcx)
100004b31: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b35: c6 44 0a 02 00              	movb	$0, 2(%rdx,%rcx)
100004b3a: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b3e: c6 44 0a 03 00              	movb	$0, 3(%rdx,%rcx)
100004b43: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b47: c6 44 0a 04 00              	movb	$0, 4(%rdx,%rcx)
100004b4c: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b50: c6 44 0a 05 00              	movb	$0, 5(%rdx,%rcx)
100004b55: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b59: c6 44 0a 06 00              	movb	$0, 6(%rdx,%rcx)
100004b5e: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b62: c6 44 0a 07 00              	movb	$0, 7(%rdx,%rcx)
100004b67: 48 83 c1 08                 	addq	$8, %rcx
100004b6b: 49 39 ce                    	cmpq	%rcx, %r14
100004b6e: 75 b0                       	jne	-80 <__ZN14ModelInterface11init_bufferEj+0x110>
100004b70: 48 85 c0                    	testq	%rax, %rax
100004b73: 74 1b                       	je	27 <__ZN14ModelInterface11init_bufferEj+0x180>
100004b75: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004b7f: 90                          	nop
100004b80: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b84: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004b88: 48 ff c1                    	incq	%rcx
100004b8b: 48 ff c8                    	decq	%rax
100004b8e: 75 f0                       	jne	-16 <__ZN14ModelInterface11init_bufferEj+0x170>
100004b90: 5b                          	popq	%rbx
100004b91: 41 5c                       	popq	%r12
100004b93: 41 5e                       	popq	%r14
100004b95: 41 5f                       	popq	%r15
100004b97: 5d                          	popq	%rbp
100004b98: c3                          	retq
100004b99: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004ba0 __ZN14ModelInterface11swap_bufferEv:
100004ba0: 55                          	pushq	%rbp
100004ba1: 48 89 e5                    	movq	%rsp, %rbp
100004ba4: 80 77 24 01                 	xorb	$1, 36(%rdi)
100004ba8: 5d                          	popq	%rbp
100004ba9: c3                          	retq
100004baa: 90                          	nop
100004bab: 90                          	nop
100004bac: 90                          	nop
100004bad: 90                          	nop
100004bae: 90                          	nop
100004baf: 90                          	nop

0000000100004bb0 __Z4ReLUPaS_j:
100004bb0: 55                          	pushq	%rbp
100004bb1: 48 89 e5                    	movq	%rsp, %rbp
100004bb4: 83 fa 04                    	cmpl	$4, %edx
100004bb7: 0f 82 88 00 00 00           	jb	136 <__Z4ReLUPaS_j+0x95>
100004bbd: 8d 42 fc                    	leal	-4(%rdx), %eax
100004bc0: 41 89 c2                    	movl	%eax, %r10d
100004bc3: 41 c1 ea 02                 	shrl	$2, %r10d
100004bc7: 41 ff c2                    	incl	%r10d
100004bca: 41 83 fa 1f                 	cmpl	$31, %r10d
100004bce: 76 24                       	jbe	36 <__Z4ReLUPaS_j+0x44>
100004bd0: 83 e0 fc                    	andl	$-4, %eax
100004bd3: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100004bd7: 48 83 c1 04                 	addq	$4, %rcx
100004bdb: 48 39 f9                    	cmpq	%rdi, %rcx
100004bde: 0f 86 78 02 00 00           	jbe	632 <__Z4ReLUPaS_j+0x2ac>
100004be4: 48 01 f8                    	addq	%rdi, %rax
100004be7: 48 83 c0 04                 	addq	$4, %rax
100004beb: 48 39 f0                    	cmpq	%rsi, %rax
100004bee: 0f 86 68 02 00 00           	jbe	616 <__Z4ReLUPaS_j+0x2ac>
100004bf4: 89 d0                       	movl	%edx, %eax
100004bf6: 45 31 c0                    	xorl	%r8d, %r8d
100004bf9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004c00: 0f b6 0e                    	movzbl	(%rsi), %ecx
100004c03: 84 c9                       	testb	%cl, %cl
100004c05: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c09: 88 0f                       	movb	%cl, (%rdi)
100004c0b: 0f b6 4e 01                 	movzbl	1(%rsi), %ecx
100004c0f: 84 c9                       	testb	%cl, %cl
100004c11: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c15: 88 4f 01                    	movb	%cl, 1(%rdi)
100004c18: 0f b6 4e 02                 	movzbl	2(%rsi), %ecx
100004c1c: 84 c9                       	testb	%cl, %cl
100004c1e: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c22: 88 4f 02                    	movb	%cl, 2(%rdi)
100004c25: 0f b6 4e 03                 	movzbl	3(%rsi), %ecx
100004c29: 84 c9                       	testb	%cl, %cl
100004c2b: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c2f: 88 4f 03                    	movb	%cl, 3(%rdi)
100004c32: 48 83 c7 04                 	addq	$4, %rdi
100004c36: 48 83 c6 04                 	addq	$4, %rsi
100004c3a: 83 c0 fc                    	addl	$-4, %eax
100004c3d: 83 f8 03                    	cmpl	$3, %eax
100004c40: 77 be                       	ja	-66 <__Z4ReLUPaS_j+0x50>
100004c42: 83 e2 03                    	andl	$3, %edx
100004c45: 85 d2                       	testl	%edx, %edx
100004c47: 0f 84 0a 02 00 00           	je	522 <__Z4ReLUPaS_j+0x2a7>
100004c4d: 8d 42 ff                    	leal	-1(%rdx), %eax
100004c50: 4c 8d 50 01                 	leaq	1(%rax), %r10
100004c54: 49 83 fa 7f                 	cmpq	$127, %r10
100004c58: 0f 86 2e 01 00 00           	jbe	302 <__Z4ReLUPaS_j+0x1dc>
100004c5e: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100004c62: 48 83 c1 01                 	addq	$1, %rcx
100004c66: 48 39 cf                    	cmpq	%rcx, %rdi
100004c69: 73 10                       	jae	16 <__Z4ReLUPaS_j+0xcb>
100004c6b: 48 01 f8                    	addq	%rdi, %rax
100004c6e: 48 83 c0 01                 	addq	$1, %rax
100004c72: 48 39 c6                    	cmpq	%rax, %rsi
100004c75: 0f 82 11 01 00 00           	jb	273 <__Z4ReLUPaS_j+0x1dc>
100004c7b: 4d 89 d0                    	movq	%r10, %r8
100004c7e: 49 83 e0 80                 	andq	$-128, %r8
100004c82: 49 8d 40 80                 	leaq	-128(%r8), %rax
100004c86: 48 89 c1                    	movq	%rax, %rcx
100004c89: 48 c1 e9 07                 	shrq	$7, %rcx
100004c8d: 48 ff c1                    	incq	%rcx
100004c90: 41 89 c9                    	movl	%ecx, %r9d
100004c93: 41 83 e1 01                 	andl	$1, %r9d
100004c97: 48 85 c0                    	testq	%rax, %rax
100004c9a: 0f 84 0f 09 00 00           	je	2319 <__Z4ReLUPaS_j+0x9ff>
100004ca0: 4c 89 c8                    	movq	%r9, %rax
100004ca3: 48 29 c8                    	subq	%rcx, %rax
100004ca6: 31 c9                       	xorl	%ecx, %ecx
100004ca8: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100004cac: 0f 1f 40 00                 	nopl	(%rax)
100004cb0: c4 e2 7d 3c 0c 0e           	vpmaxsb	(%rsi,%rcx), %ymm0, %ymm1
100004cb6: c4 e2 7d 3c 54 0e 20        	vpmaxsb	32(%rsi,%rcx), %ymm0, %ymm2
100004cbd: c4 e2 7d 3c 5c 0e 40        	vpmaxsb	64(%rsi,%rcx), %ymm0, %ymm3
100004cc4: c4 e2 7d 3c 64 0e 60        	vpmaxsb	96(%rsi,%rcx), %ymm0, %ymm4
100004ccb: c5 fe 7f 0c 0f              	vmovdqu	%ymm1, (%rdi,%rcx)
100004cd0: c5 fe 7f 54 0f 20           	vmovdqu	%ymm2, 32(%rdi,%rcx)
100004cd6: c5 fe 7f 5c 0f 40           	vmovdqu	%ymm3, 64(%rdi,%rcx)
100004cdc: c5 fe 7f 64 0f 60           	vmovdqu	%ymm4, 96(%rdi,%rcx)
100004ce2: c4 e2 7d 3c 8c 0e 80 00 00 00       	vpmaxsb	128(%rsi,%rcx), %ymm0, %ymm1
100004cec: c4 e2 7d 3c 94 0e a0 00 00 00       	vpmaxsb	160(%rsi,%rcx), %ymm0, %ymm2
100004cf6: c4 e2 7d 3c 9c 0e c0 00 00 00       	vpmaxsb	192(%rsi,%rcx), %ymm0, %ymm3
100004d00: c4 e2 7d 3c a4 0e e0 00 00 00       	vpmaxsb	224(%rsi,%rcx), %ymm0, %ymm4
100004d0a: c5 fe 7f 8c 0f 80 00 00 00  	vmovdqu	%ymm1, 128(%rdi,%rcx)
100004d13: c5 fe 7f 94 0f a0 00 00 00  	vmovdqu	%ymm2, 160(%rdi,%rcx)
100004d1c: c5 fe 7f 9c 0f c0 00 00 00  	vmovdqu	%ymm3, 192(%rdi,%rcx)
100004d25: c5 fe 7f a4 0f e0 00 00 00  	vmovdqu	%ymm4, 224(%rdi,%rcx)
100004d2e: 48 81 c1 00 01 00 00        	addq	$256, %rcx
100004d35: 48 83 c0 02                 	addq	$2, %rax
100004d39: 0f 85 71 ff ff ff           	jne	-143 <__Z4ReLUPaS_j+0x100>
100004d3f: 4d 85 c9                    	testq	%r9, %r9
100004d42: 74 36                       	je	54 <__Z4ReLUPaS_j+0x1ca>
100004d44: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100004d48: c4 e2 7d 3c 0c 0e           	vpmaxsb	(%rsi,%rcx), %ymm0, %ymm1
100004d4e: c4 e2 7d 3c 54 0e 20        	vpmaxsb	32(%rsi,%rcx), %ymm0, %ymm2
100004d55: c4 e2 7d 3c 5c 0e 40        	vpmaxsb	64(%rsi,%rcx), %ymm0, %ymm3
100004d5c: c4 e2 7d 3c 44 0e 60        	vpmaxsb	96(%rsi,%rcx), %ymm0, %ymm0
100004d63: c5 fe 7f 0c 0f              	vmovdqu	%ymm1, (%rdi,%rcx)
100004d68: c5 fe 7f 54 0f 20           	vmovdqu	%ymm2, 32(%rdi,%rcx)
100004d6e: c5 fe 7f 5c 0f 40           	vmovdqu	%ymm3, 64(%rdi,%rcx)
100004d74: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
100004d7a: 4d 39 c2                    	cmpq	%r8, %r10
100004d7d: 0f 84 d4 00 00 00           	je	212 <__Z4ReLUPaS_j+0x2a7>
100004d83: 44 29 c2                    	subl	%r8d, %edx
100004d86: 4c 01 c6                    	addq	%r8, %rsi
100004d89: 4c 01 c7                    	addq	%r8, %rdi
100004d8c: 44 8d 42 ff                 	leal	-1(%rdx), %r8d
100004d90: f6 c2 07                    	testb	$7, %dl
100004d93: 74 38                       	je	56 <__Z4ReLUPaS_j+0x21d>
100004d95: 41 89 d2                    	movl	%edx, %r10d
100004d98: 41 83 e2 07                 	andl	$7, %r10d
100004d9c: 45 31 c9                    	xorl	%r9d, %r9d
100004d9f: 31 c9                       	xorl	%ecx, %ecx
100004da1: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004dab: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004db0: 0f b6 04 0e                 	movzbl	(%rsi,%rcx), %eax
100004db4: 84 c0                       	testb	%al, %al
100004db6: 41 0f 48 c1                 	cmovsl	%r9d, %eax
100004dba: 88 04 0f                    	movb	%al, (%rdi,%rcx)
100004dbd: 48 ff c1                    	incq	%rcx
100004dc0: 41 39 ca                    	cmpl	%ecx, %r10d
100004dc3: 75 eb                       	jne	-21 <__Z4ReLUPaS_j+0x200>
100004dc5: 29 ca                       	subl	%ecx, %edx
100004dc7: 48 01 ce                    	addq	%rcx, %rsi
100004dca: 48 01 cf                    	addq	%rcx, %rdi
100004dcd: 41 83 f8 07                 	cmpl	$7, %r8d
100004dd1: 0f 82 80 00 00 00           	jb	128 <__Z4ReLUPaS_j+0x2a7>
100004dd7: 41 89 d0                    	movl	%edx, %r8d
100004dda: 31 c9                       	xorl	%ecx, %ecx
100004ddc: 31 d2                       	xorl	%edx, %edx
100004dde: 66 90                       	nop
100004de0: 0f b6 04 16                 	movzbl	(%rsi,%rdx), %eax
100004de4: 84 c0                       	testb	%al, %al
100004de6: 0f 48 c1                    	cmovsl	%ecx, %eax
100004de9: 88 04 17                    	movb	%al, (%rdi,%rdx)
100004dec: 0f b6 44 16 01              	movzbl	1(%rsi,%rdx), %eax
100004df1: 84 c0                       	testb	%al, %al
100004df3: 0f 48 c1                    	cmovsl	%ecx, %eax
100004df6: 88 44 17 01                 	movb	%al, 1(%rdi,%rdx)
100004dfa: 0f b6 44 16 02              	movzbl	2(%rsi,%rdx), %eax
100004dff: 84 c0                       	testb	%al, %al
100004e01: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e04: 88 44 17 02                 	movb	%al, 2(%rdi,%rdx)
100004e08: 0f b6 44 16 03              	movzbl	3(%rsi,%rdx), %eax
100004e0d: 84 c0                       	testb	%al, %al
100004e0f: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e12: 88 44 17 03                 	movb	%al, 3(%rdi,%rdx)
100004e16: 0f b6 44 16 04              	movzbl	4(%rsi,%rdx), %eax
100004e1b: 84 c0                       	testb	%al, %al
100004e1d: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e20: 88 44 17 04                 	movb	%al, 4(%rdi,%rdx)
100004e24: 0f b6 44 16 05              	movzbl	5(%rsi,%rdx), %eax
100004e29: 84 c0                       	testb	%al, %al
100004e2b: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e2e: 88 44 17 05                 	movb	%al, 5(%rdi,%rdx)
100004e32: 0f b6 44 16 06              	movzbl	6(%rsi,%rdx), %eax
100004e37: 84 c0                       	testb	%al, %al
100004e39: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e3c: 88 44 17 06                 	movb	%al, 6(%rdi,%rdx)
100004e40: 0f b6 44 16 07              	movzbl	7(%rsi,%rdx), %eax
100004e45: 84 c0                       	testb	%al, %al
100004e47: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e4a: 88 44 17 07                 	movb	%al, 7(%rdi,%rdx)
100004e4e: 48 83 c2 08                 	addq	$8, %rdx
100004e52: 41 39 d0                    	cmpl	%edx, %r8d
100004e55: 75 89                       	jne	-119 <__Z4ReLUPaS_j+0x230>
100004e57: 5d                          	popq	%rbp
100004e58: c5 f8 77                    	vzeroupper
100004e5b: c3                          	retq
100004e5c: 45 89 d0                    	movl	%r10d, %r8d
100004e5f: 41 83 e0 e0                 	andl	$-32, %r8d
100004e63: 49 8d 40 e0                 	leaq	-32(%r8), %rax
100004e67: 48 89 c1                    	movq	%rax, %rcx
100004e6a: 48 c1 e9 05                 	shrq	$5, %rcx
100004e6e: 48 ff c1                    	incq	%rcx
100004e71: 41 89 c9                    	movl	%ecx, %r9d
100004e74: 41 83 e1 01                 	andl	$1, %r9d
100004e78: 48 85 c0                    	testq	%rax, %rax
100004e7b: 0f 84 3e 07 00 00           	je	1854 <__Z4ReLUPaS_j+0xa0f>
100004e81: 4c 89 c8                    	movq	%r9, %rax
100004e84: 48 29 c8                    	subq	%rcx, %rax
100004e87: 31 c9                       	xorl	%ecx, %ecx
100004e89: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004e90: c5 7a 6f 34 0e              	vmovdqu	(%rsi,%rcx), %xmm14
100004e95: c5 7a 6f 7c 0e 10           	vmovdqu	16(%rsi,%rcx), %xmm15
100004e9b: c5 fa 6f 54 0e 20           	vmovdqu	32(%rsi,%rcx), %xmm2
100004ea1: c5 fa 6f 5c 0e 30           	vmovdqu	48(%rsi,%rcx), %xmm3
100004ea7: c5 79 6f 1d 71 22 00 00     	vmovdqa	8817(%rip), %xmm11
100004eaf: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100004eb4: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
100004eb9: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100004ebd: c5 79 6f 05 6b 22 00 00     	vmovdqa	8811(%rip), %xmm8
100004ec5: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
100004eca: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
100004ecf: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004ed3: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
100004ed9: c5 fa 6f 64 0e 70           	vmovdqu	112(%rsi,%rcx), %xmm4
100004edf: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100004ee4: c4 e3 fd 00 6c 0e 60 4e     	vpermq	$78, 96(%rsi,%rcx), %ymm5
100004eec: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004ef2: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004ef7: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
100004efb: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100004f01: c5 fa 6f 74 0e 50           	vmovdqu	80(%rsi,%rcx), %xmm6
100004f07: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
100004f0c: c4 63 fd 00 4c 0e 40 4e     	vpermq	$78, 64(%rsi,%rcx), %ymm9
100004f14: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
100004f1a: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
100004f1f: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100004f24: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004f2a: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100004f30: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004f36: c5 79 6f 05 02 22 00 00     	vmovdqa	8706(%rip), %xmm8
100004f3e: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100004f43: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100004f48: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
100004f4c: c5 79 6f 1d fc 21 00 00     	vmovdqa	8700(%rip), %xmm11
100004f54: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
100004f59: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100004f5e: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
100004f62: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
100004f68: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
100004f6d: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
100004f72: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004f76: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004f7c: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
100004f81: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
100004f86: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100004f8a: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004f90: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004f96: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
100004f9c: c5 79 6f 1d bc 21 00 00     	vmovdqa	8636(%rip), %xmm11
100004fa4: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100004fa9: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
100004fae: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
100004fb2: c5 f9 6f 0d b6 21 00 00     	vmovdqa	8630(%rip), %xmm1
100004fba: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
100004fbe: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100004fc3: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100004fc8: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004fcc: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100004fd2: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100004fd7: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004fdc: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004fe0: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004fe6: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
100004feb: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100004ff0: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100004ff4: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004ffa: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005000: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100005006: c5 f9 6f 0d 72 21 00 00     	vmovdqa	8562(%rip), %xmm1
10000500e: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100005013: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100005018: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000501c: c5 f9 6f 15 6c 21 00 00     	vmovdqa	8556(%rip), %xmm2
100005024: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100005028: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
10000502d: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100005032: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005036: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
10000503c: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100005041: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100005046: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000504a: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005050: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
100005055: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
10000505a: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
10000505e: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100005064: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
10000506a: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
100005070: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
100005075: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
10000507a: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
10000507f: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
100005084: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100005089: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
10000508d: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005091: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005095: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100005099: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
10000509d: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000050a1: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000050a5: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000050a9: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
1000050af: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
1000050b5: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
1000050bb: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
1000050c1: c5 fe 7f 4c 0f 40           	vmovdqu	%ymm1, 64(%rdi,%rcx)
1000050c7: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
1000050cd: c5 fe 7f 6c 0f 20           	vmovdqu	%ymm5, 32(%rdi,%rcx)
1000050d3: c5 fe 7f 14 0f              	vmovdqu	%ymm2, (%rdi,%rcx)
1000050d8: c5 7a 6f a4 0e 80 00 00 00  	vmovdqu	128(%rsi,%rcx), %xmm12
1000050e1: c5 7a 6f ac 0e 90 00 00 00  	vmovdqu	144(%rsi,%rcx), %xmm13
1000050ea: c5 7a 6f b4 0e a0 00 00 00  	vmovdqu	160(%rsi,%rcx), %xmm14
1000050f3: c5 fa 6f 9c 0e b0 00 00 00  	vmovdqu	176(%rsi,%rcx), %xmm3
1000050fc: c5 f9 6f 05 1c 20 00 00     	vmovdqa	8220(%rip), %xmm0
100005104: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
100005109: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
10000510e: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100005112: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100005116: c5 f9 6f 05 12 20 00 00     	vmovdqa	8210(%rip), %xmm0
10000511e: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100005123: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
100005128: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
10000512c: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005130: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
100005136: c5 fa 6f a4 0e f0 00 00 00  	vmovdqu	240(%rsi,%rcx), %xmm4
10000513f: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100005144: c4 e3 fd 00 ac 0e e0 00 00 00 4e    	vpermq	$78, 224(%rsi,%rcx), %ymm5
10000514f: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005155: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
10000515a: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000515e: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
100005164: c5 fa 6f b4 0e d0 00 00 00  	vmovdqu	208(%rsi,%rcx), %xmm6
10000516d: c4 e3 fd 00 bc 0e c0 00 00 00 4e    	vpermq	$78, 192(%rsi,%rcx), %ymm7
100005178: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
10000517d: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
100005183: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
100005188: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
10000518c: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005192: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
100005198: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
10000519e: c5 79 6f 3d 9a 1f 00 00     	vmovdqa	8090(%rip), %xmm15
1000051a6: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
1000051ab: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
1000051b0: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
1000051b4: c5 f9 6f 05 94 1f 00 00     	vmovdqa	8084(%rip), %xmm0
1000051bc: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000051c1: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000051c6: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000051ca: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
1000051d0: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
1000051d5: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
1000051da: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000051de: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000051e4: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
1000051e9: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
1000051ee: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
1000051f2: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000051f8: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000051fe: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005204: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005209: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000520e: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100005212: c5 f9 6f 05 56 1f 00 00     	vmovdqa	8022(%rip), %xmm0
10000521a: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000521f: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005224: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005228: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
10000522e: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005233: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100005238: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000523c: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005241: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005246: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
10000524a: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005250: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005256: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
10000525c: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
100005262: c5 79 6f 3d 16 1f 00 00     	vmovdqa	7958(%rip), %xmm15
10000526a: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
10000526f: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
100005274: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005278: c5 f9 6f 05 10 1f 00 00     	vmovdqa	7952(%rip), %xmm0
100005280: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
100005285: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
10000528a: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000528e: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
100005294: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
100005299: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
10000529e: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000052a2: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
1000052a7: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
1000052ac: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000052b0: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000052b6: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
1000052bc: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
1000052c2: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
1000052c8: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
1000052cd: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
1000052d2: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
1000052d7: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000052dc: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
1000052e0: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000052e4: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
1000052e8: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
1000052ec: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000052f0: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000052f4: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000052f8: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000052fc: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100005302: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100005308: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
10000530e: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005314: c5 fe 7f 8c 0f c0 00 00 00  	vmovdqu	%ymm1, 192(%rdi,%rcx)
10000531d: c5 fe 7f 84 0f e0 00 00 00  	vmovdqu	%ymm0, 224(%rdi,%rcx)
100005326: c5 fe 7f 9c 0f a0 00 00 00  	vmovdqu	%ymm3, 160(%rdi,%rcx)
10000532f: c5 fe 7f 94 0f 80 00 00 00  	vmovdqu	%ymm2, 128(%rdi,%rcx)
100005338: 48 81 c1 00 01 00 00        	addq	$256, %rcx
10000533f: 48 83 c0 02                 	addq	$2, %rax
100005343: 0f 85 47 fb ff ff           	jne	-1209 <__Z4ReLUPaS_j+0x2e0>
100005349: 4d 85 c9                    	testq	%r9, %r9
10000534c: 0f 84 3e 02 00 00           	je	574 <__Z4ReLUPaS_j+0x9e0>
100005352: c5 7a 6f 14 0e              	vmovdqu	(%rsi,%rcx), %xmm10
100005357: c5 7a 6f 5c 0e 10           	vmovdqu	16(%rsi,%rcx), %xmm11
10000535d: c5 7a 6f 64 0e 20           	vmovdqu	32(%rsi,%rcx), %xmm12
100005363: c5 7a 6f 6c 0e 30           	vmovdqu	48(%rsi,%rcx), %xmm13
100005369: c5 f9 6f 35 af 1d 00 00     	vmovdqa	7599(%rip), %xmm6
100005371: c4 e2 11 00 e6              	vpshufb	%xmm6, %xmm13, %xmm4
100005376: c4 e2 19 00 ee              	vpshufb	%xmm6, %xmm12, %xmm5
10000537b: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000537f: c5 f9 6f 05 a9 1d 00 00     	vmovdqa	7593(%rip), %xmm0
100005387: c4 e2 21 00 e8              	vpshufb	%xmm0, %xmm11, %xmm5
10000538c: c4 e2 29 00 f8              	vpshufb	%xmm0, %xmm10, %xmm7
100005391: c5 c1 62 ed                 	vpunpckldq	%xmm5, %xmm7, %xmm5
100005395: c4 63 51 02 c4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm8
10000539b: c5 7a 6f 74 0e 70           	vmovdqu	112(%rsi,%rcx), %xmm14
1000053a1: c4 e2 09 00 fe              	vpshufb	%xmm6, %xmm14, %xmm7
1000053a6: c4 e3 fd 00 6c 0e 60 4e     	vpermq	$78, 96(%rsi,%rcx), %ymm5
1000053ae: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
1000053b4: c4 e2 51 00 f6              	vpshufb	%xmm6, %xmm5, %xmm6
1000053b9: c5 c9 62 f7                 	vpunpckldq	%xmm7, %xmm6, %xmm6
1000053bd: c4 63 7d 38 ce 01           	vinserti128	$1, %xmm6, %ymm0, %ymm9
1000053c3: c5 fa 6f 74 0e 50           	vmovdqu	80(%rsi,%rcx), %xmm6
1000053c9: c4 e2 49 00 c8              	vpshufb	%xmm0, %xmm6, %xmm1
1000053ce: c4 e3 fd 00 7c 0e 40 4e     	vpermq	$78, 64(%rsi,%rcx), %ymm7
1000053d6: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000053dc: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
1000053e1: c5 f9 62 c1                 	vpunpckldq	%xmm1, %xmm0, %xmm0
1000053e5: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000053eb: c4 c3 7d 02 c1 c0           	vpblendd	$192, %ymm9, %ymm0, %ymm0
1000053f1: c4 63 3d 02 c0 f0           	vpblendd	$240, %ymm0, %ymm8, %ymm8
1000053f7: c5 f9 6f 05 41 1d 00 00     	vmovdqa	7489(%rip), %xmm0
1000053ff: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100005404: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005409: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000540d: c5 f9 6f 15 3b 1d 00 00     	vmovdqa	7483(%rip), %xmm2
100005415: c4 e2 21 00 da              	vpshufb	%xmm2, %xmm11, %xmm3
10000541a: c4 e2 29 00 e2              	vpshufb	%xmm2, %xmm10, %xmm4
10000541f: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005423: c4 e3 61 02 c9 0c           	vpblendd	$12, %xmm1, %xmm3, %xmm1
100005429: c4 e2 09 00 d8              	vpshufb	%xmm0, %xmm14, %xmm3
10000542e: c4 e2 51 00 c0              	vpshufb	%xmm0, %xmm5, %xmm0
100005433: c5 f9 62 c3                 	vpunpckldq	%xmm3, %xmm0, %xmm0
100005437: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000543d: c4 e2 49 00 da              	vpshufb	%xmm2, %xmm6, %xmm3
100005442: c4 e2 41 00 d2              	vpshufb	%xmm2, %xmm7, %xmm2
100005447: c5 e9 62 d3                 	vpunpckldq	%xmm3, %xmm2, %xmm2
10000544b: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005451: c4 e3 6d 02 c0 c0           	vpblendd	$192, %ymm0, %ymm2, %ymm0
100005457: c4 63 75 02 c8 f0           	vpblendd	$240, %ymm0, %ymm1, %ymm9
10000545d: c5 f9 6f 05 fb 1c 00 00     	vmovdqa	7419(%rip), %xmm0
100005465: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000546a: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
10000546f: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005473: c5 f9 6f 15 f5 1c 00 00     	vmovdqa	7413(%rip), %xmm2
10000547b: c4 e2 21 00 da              	vpshufb	%xmm2, %xmm11, %xmm3
100005480: c4 e2 29 00 e2              	vpshufb	%xmm2, %xmm10, %xmm4
100005485: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005489: c4 e3 61 02 c9 0c           	vpblendd	$12, %xmm1, %xmm3, %xmm1
10000548f: c4 e2 09 00 d8              	vpshufb	%xmm0, %xmm14, %xmm3
100005494: c4 e2 51 00 c0              	vpshufb	%xmm0, %xmm5, %xmm0
100005499: c5 f9 62 c3                 	vpunpckldq	%xmm3, %xmm0, %xmm0
10000549d: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000054a3: c4 e2 49 00 da              	vpshufb	%xmm2, %xmm6, %xmm3
1000054a8: c4 e2 41 00 d2              	vpshufb	%xmm2, %xmm7, %xmm2
1000054ad: c5 e9 62 d3                 	vpunpckldq	%xmm3, %xmm2, %xmm2
1000054b1: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000054b7: c4 e3 6d 02 c0 c0           	vpblendd	$192, %ymm0, %ymm2, %ymm0
1000054bd: c4 63 75 02 f8 f0           	vpblendd	$240, %ymm0, %ymm1, %ymm15
1000054c3: c5 f9 6f 0d b5 1c 00 00     	vmovdqa	7349(%rip), %xmm1
1000054cb: c4 e2 11 00 d1              	vpshufb	%xmm1, %xmm13, %xmm2
1000054d0: c4 e2 19 00 d9              	vpshufb	%xmm1, %xmm12, %xmm3
1000054d5: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000054d9: c5 f9 6f 1d af 1c 00 00     	vmovdqa	7343(%rip), %xmm3
1000054e1: c4 e2 21 00 e3              	vpshufb	%xmm3, %xmm11, %xmm4
1000054e6: c4 e2 29 00 c3              	vpshufb	%xmm3, %xmm10, %xmm0
1000054eb: c5 f9 62 c4                 	vpunpckldq	%xmm4, %xmm0, %xmm0
1000054ef: c4 e3 79 02 c2 0c           	vpblendd	$12, %xmm2, %xmm0, %xmm0
1000054f5: c4 e2 09 00 d1              	vpshufb	%xmm1, %xmm14, %xmm2
1000054fa: c4 e2 51 00 c9              	vpshufb	%xmm1, %xmm5, %xmm1
1000054ff: c5 f1 62 ca                 	vpunpckldq	%xmm2, %xmm1, %xmm1
100005503: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005509: c4 e2 49 00 d3              	vpshufb	%xmm3, %xmm6, %xmm2
10000550e: c4 e2 41 00 db              	vpshufb	%xmm3, %xmm7, %xmm3
100005513: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005517: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
10000551d: c4 e3 6d 02 c9 c0           	vpblendd	$192, %ymm1, %ymm2, %ymm1
100005523: c4 e3 7d 02 c1 f0           	vpblendd	$240, %ymm1, %ymm0, %ymm0
100005529: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
10000552d: c4 e2 3d 3c d1              	vpmaxsb	%ymm1, %ymm8, %ymm2
100005532: c4 e2 35 3c d9              	vpmaxsb	%ymm1, %ymm9, %ymm3
100005537: c4 e2 05 3c e1              	vpmaxsb	%ymm1, %ymm15, %ymm4
10000553c: c4 e2 7d 3c c1              	vpmaxsb	%ymm1, %ymm0, %ymm0
100005541: c5 ed 60 cb                 	vpunpcklbw	%ymm3, %ymm2, %ymm1
100005545: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005549: c5 dd 60 d8                 	vpunpcklbw	%ymm0, %ymm4, %ymm3
10000554d: c5 dd 68 c0                 	vpunpckhbw	%ymm0, %ymm4, %ymm0
100005551: c5 f5 61 e3                 	vpunpcklwd	%ymm3, %ymm1, %ymm4
100005555: c5 f5 69 cb                 	vpunpckhwd	%ymm3, %ymm1, %ymm1
100005559: c5 ed 61 d8                 	vpunpcklwd	%ymm0, %ymm2, %ymm3
10000555d: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100005561: c4 e3 5d 38 d1 01           	vinserti128	$1, %xmm1, %ymm4, %ymm2
100005567: c4 e3 65 38 e8 01           	vinserti128	$1, %xmm0, %ymm3, %ymm5
10000556d: c4 e3 5d 46 c9 31           	vperm2i128	$49, %ymm1, %ymm4, %ymm1
100005573: c4 e3 65 46 c0 31           	vperm2i128	$49, %ymm0, %ymm3, %ymm0
100005579: c5 fe 7f 4c 0f 40           	vmovdqu	%ymm1, 64(%rdi,%rcx)
10000557f: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
100005585: c5 fe 7f 6c 0f 20           	vmovdqu	%ymm5, 32(%rdi,%rcx)
10000558b: c5 fe 7f 14 0f              	vmovdqu	%ymm2, (%rdi,%rcx)
100005590: 4a 8d 34 86                 	leaq	(%rsi,%r8,4), %rsi
100005594: 4a 8d 3c 87                 	leaq	(%rdi,%r8,4), %rdi
100005598: 4d 39 d0                    	cmpq	%r10, %r8
10000559b: 0f 84 a1 f6 ff ff           	je	-2399 <__Z4ReLUPaS_j+0x92>
1000055a1: 41 c1 e0 02                 	shll	$2, %r8d
1000055a5: 89 d0                       	movl	%edx, %eax
1000055a7: 44 29 c0                    	subl	%r8d, %eax
1000055aa: e9 47 f6 ff ff              	jmp	-2489 <__Z4ReLUPaS_j+0x46>
1000055af: 31 c9                       	xorl	%ecx, %ecx
1000055b1: 4d 85 c9                    	testq	%r9, %r9
1000055b4: 0f 85 8a f7 ff ff           	jne	-2166 <__Z4ReLUPaS_j+0x194>
1000055ba: e9 bb f7 ff ff              	jmp	-2117 <__Z4ReLUPaS_j+0x1ca>
1000055bf: 31 c9                       	xorl	%ecx, %ecx
1000055c1: 4d 85 c9                    	testq	%r9, %r9
1000055c4: 0f 85 88 fd ff ff           	jne	-632 <__Z4ReLUPaS_j+0x7a2>
1000055ca: eb c4                       	jmp	-60 <__Z4ReLUPaS_j+0x9e0>
1000055cc: 90                          	nop
1000055cd: 90                          	nop
1000055ce: 90                          	nop
1000055cf: 90                          	nop

00000001000055d0 __ZN11LineNetworkC2Ev:
1000055d0: 55                          	pushq	%rbp
1000055d1: 48 89 e5                    	movq	%rsp, %rbp
1000055d4: 41 56                       	pushq	%r14
1000055d6: 53                          	pushq	%rbx
1000055d7: 48 89 fb                    	movq	%rdi, %rbx
1000055da: e8 f1 f2 ff ff              	callq	-3343 <__ZN14ModelInterfaceC2Ev>
1000055df: 48 8d 05 0a 3b 00 00        	leaq	15114(%rip), %rax
1000055e6: 48 89 03                    	movq	%rax, (%rbx)
1000055e9: 48 89 df                    	movq	%rbx, %rdi
1000055ec: be 00 00 08 00              	movl	$524288, %esi
1000055f1: e8 1a f4 ff ff              	callq	-3046 <__ZN14ModelInterface11init_bufferEj>
1000055f6: c7 43 20 00 04 a2 01        	movl	$27395072, 32(%rbx)
1000055fd: c5 f8 28 05 9b 1b 00 00     	vmovaps	7067(%rip), %xmm0
100005605: c5 f8 11 43 08              	vmovups	%xmm0, 8(%rbx)
10000560a: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
100005614: 48 89 43 18                 	movq	%rax, 24(%rbx)
100005618: 5b                          	popq	%rbx
100005619: 41 5e                       	popq	%r14
10000561b: 5d                          	popq	%rbp
10000561c: c3                          	retq
10000561d: 49 89 c6                    	movq	%rax, %r14
100005620: 48 89 df                    	movq	%rbx, %rdi
100005623: e8 e8 f2 ff ff              	callq	-3352 <__ZN14ModelInterfaceD2Ev>
100005628: 4c 89 f7                    	movq	%r14, %rdi
10000562b: e8 a8 17 00 00              	callq	6056 <dyld_stub_binder+0x100006dd8>
100005630: 0f 0b                       	ud2
100005632: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000563c: 0f 1f 40 00                 	nopl	(%rax)

0000000100005640 __ZN11LineNetworkC1Ev:
100005640: 55                          	pushq	%rbp
100005641: 48 89 e5                    	movq	%rsp, %rbp
100005644: 41 56                       	pushq	%r14
100005646: 53                          	pushq	%rbx
100005647: 48 89 fb                    	movq	%rdi, %rbx
10000564a: e8 81 f2 ff ff              	callq	-3455 <__ZN14ModelInterfaceC2Ev>
10000564f: 48 8d 05 9a 3a 00 00        	leaq	15002(%rip), %rax
100005656: 48 89 03                    	movq	%rax, (%rbx)
100005659: 48 89 df                    	movq	%rbx, %rdi
10000565c: be 00 00 08 00              	movl	$524288, %esi
100005661: e8 aa f3 ff ff              	callq	-3158 <__ZN14ModelInterface11init_bufferEj>
100005666: c7 43 20 00 04 a2 01        	movl	$27395072, 32(%rbx)
10000566d: c5 f8 28 05 2b 1b 00 00     	vmovaps	6955(%rip), %xmm0
100005675: c5 f8 11 43 08              	vmovups	%xmm0, 8(%rbx)
10000567a: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
100005684: 48 89 43 18                 	movq	%rax, 24(%rbx)
100005688: 5b                          	popq	%rbx
100005689: 41 5e                       	popq	%r14
10000568b: 5d                          	popq	%rbp
10000568c: c3                          	retq
10000568d: 49 89 c6                    	movq	%rax, %r14
100005690: 48 89 df                    	movq	%rbx, %rdi
100005693: e8 78 f2 ff ff              	callq	-3464 <__ZN14ModelInterfaceD2Ev>
100005698: 4c 89 f7                    	movq	%r14, %rdi
10000569b: e8 38 17 00 00              	callq	5944 <dyld_stub_binder+0x100006dd8>
1000056a0: 0f 0b                       	ud2
1000056a2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000056ac: 0f 1f 40 00                 	nopl	(%rax)

00000001000056b0 __ZN11LineNetwork7forwardEv:
1000056b0: 55                          	pushq	%rbp
1000056b1: 48 89 e5                    	movq	%rsp, %rbp
1000056b4: 41 57                       	pushq	%r15
1000056b6: 41 56                       	pushq	%r14
1000056b8: 41 55                       	pushq	%r13
1000056ba: 41 54                       	pushq	%r12
1000056bc: 53                          	pushq	%rbx
1000056bd: 48 83 ec 58                 	subq	$88, %rsp
1000056c1: 49 89 ff                    	movq	%rdi, %r15
1000056c4: e8 27 f3 ff ff              	callq	-3289 <__ZN14ModelInterface13output_bufferEv>
1000056c9: 49 89 c6                    	movq	%rax, %r14
1000056cc: 4c 89 ff                    	movq	%r15, %rdi
1000056cf: e8 0c f3 ff ff              	callq	-3316 <__ZN14ModelInterface12input_bufferEv>
1000056d4: 48 8d 15 85 1c 00 00        	leaq	7301(%rip), %rdx
1000056db: 48 8d 0d c6 1c 00 00        	leaq	7366(%rip), %rcx
1000056e2: 4c 89 f7                    	movq	%r14, %rdi
1000056e5: 48 89 c6                    	movq	%rax, %rsi
1000056e8: 41 b8 37 00 00 00           	movl	$55, %r8d
1000056ee: e8 6d 05 00 00              	callq	1389 <__ZN11LineNetwork7forwardEv+0x5b0>
1000056f3: 4c 89 ff                    	movq	%r15, %rdi
1000056f6: e8 a5 f4 ff ff              	callq	-2907 <__ZN14ModelInterface11swap_bufferEv>
1000056fb: 4c 89 ff                    	movq	%r15, %rdi
1000056fe: e8 ed f2 ff ff              	callq	-3347 <__ZN14ModelInterface13output_bufferEv>
100005703: 49 89 c6                    	movq	%rax, %r14
100005706: 4c 89 ff                    	movq	%r15, %rdi
100005709: e8 d2 f2 ff ff              	callq	-3374 <__ZN14ModelInterface12input_bufferEv>
10000570e: 4c 89 f7                    	movq	%r14, %rdi
100005711: 48 89 c6                    	movq	%rax, %rsi
100005714: ba 00 00 08 00              	movl	$524288, %edx
100005719: e8 92 f4 ff ff              	callq	-2926 <__Z4ReLUPaS_j>
10000571e: 4c 89 ff                    	movq	%r15, %rdi
100005721: e8 7a f4 ff ff              	callq	-2950 <__ZN14ModelInterface11swap_bufferEv>
100005726: 4c 89 ff                    	movq	%r15, %rdi
100005729: e8 c2 f2 ff ff              	callq	-3390 <__ZN14ModelInterface13output_bufferEv>
10000572e: 49 89 c5                    	movq	%rax, %r13
100005731: 4c 89 7d 88                 	movq	%r15, -120(%rbp)
100005735: 4c 89 ff                    	movq	%r15, %rdi
100005738: e8 a3 f2 ff ff              	callq	-3421 <__ZN14ModelInterface12input_bufferEv>
10000573d: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100005741: 31 c0                       	xorl	%eax, %eax
100005743: eb 23                       	jmp	35 <__ZN11LineNetwork7forwardEv+0xb8>
100005745: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000574f: 90                          	nop
100005750: 48 8b 45 c0                 	movq	-64(%rbp), %rax
100005754: 48 ff c0                    	incq	%rax
100005757: 4c 8b 6d b8                 	movq	-72(%rbp), %r13
10000575b: 49 ff c5                    	incq	%r13
10000575e: 48 83 f8 08                 	cmpq	$8, %rax
100005762: 0f 84 02 01 00 00           	je	258 <__ZN11LineNetwork7forwardEv+0x1ba>
100005768: 48 89 45 c0                 	movq	%rax, -64(%rbp)
10000576c: 48 8d 04 c5 00 00 00 00     	leaq	(,%rax,8), %rax
100005774: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005778: 48 8d 05 31 1c 00 00        	leaq	7217(%rip), %rax
10000577f: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005783: 48 89 55 90                 	movq	%rdx, -112(%rbp)
100005787: 48 8d 54 08 18              	leaq	24(%rax,%rcx), %rdx
10000578c: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005790: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
100005794: 48 8d 44 08 30              	leaq	48(%rax,%rcx), %rax
100005799: 48 89 45 a0                 	movq	%rax, -96(%rbp)
10000579d: 4c 89 6d b8                 	movq	%r13, -72(%rbp)
1000057a1: 4c 8b 7d c8                 	movq	-56(%rbp), %r15
1000057a5: 31 c0                       	xorl	%eax, %eax
1000057a7: eb 25                       	jmp	37 <__ZN11LineNetwork7forwardEv+0x11e>
1000057a9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
1000057b0: 4c 8b 7d a8                 	movq	-88(%rbp), %r15
1000057b4: 49 81 c7 00 10 00 00        	addq	$4096, %r15
1000057bb: 49 81 c5 00 04 00 00        	addq	$1024, %r13
1000057c2: 48 8b 45 b0                 	movq	-80(%rbp), %rax
1000057c6: 48 3d fd 00 00 00           	cmpq	$253, %rax
1000057cc: 73 82                       	jae	-126 <__ZN11LineNetwork7forwardEv+0xa0>
1000057ce: 48 83 c0 02                 	addq	$2, %rax
1000057d2: 48 89 45 b0                 	movq	%rax, -80(%rbp)
1000057d6: 4c 89 7d a8                 	movq	%r15, -88(%rbp)
1000057da: 31 db                       	xorl	%ebx, %ebx
1000057dc: eb 18                       	jmp	24 <__ZN11LineNetwork7forwardEv+0x146>
1000057de: 66 90                       	nop
1000057e0: 41 88 44 9d 00              	movb	%al, (%r13,%rbx,4)
1000057e5: 48 83 c3 02                 	addq	$2, %rbx
1000057e9: 49 83 c7 10                 	addq	$16, %r15
1000057ed: 48 81 fb fd 00 00 00        	cmpq	$253, %rbx
1000057f4: 73 ba                       	jae	-70 <__ZN11LineNetwork7forwardEv+0x100>
1000057f6: 4c 89 ff                    	movq	%r15, %rdi
1000057f9: 48 8b 75 90                 	movq	-112(%rbp), %rsi
1000057fd: e8 1e 12 00 00              	callq	4638 <__ZN11LineNetwork7forwardEv+0x1370>
100005802: 41 89 c6                    	movl	%eax, %r14d
100005805: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
10000580c: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100005810: e8 0b 12 00 00              	callq	4619 <__ZN11LineNetwork7forwardEv+0x1370>
100005815: 41 89 c4                    	movl	%eax, %r12d
100005818: 45 01 f4                    	addl	%r14d, %r12d
10000581b: 49 8d bf 00 10 00 00        	leaq	4096(%r15), %rdi
100005822: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005826: e8 f5 11 00 00              	callq	4597 <__ZN11LineNetwork7forwardEv+0x1370>
10000582b: 44 01 e0                    	addl	%r12d, %eax
10000582e: 48 8d 0d bb 1d 00 00        	leaq	7611(%rip), %rcx
100005835: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
100005839: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
10000583d: 01 c1                       	addl	%eax, %ecx
10000583f: 6b c9 37                    	imull	$55, %ecx, %ecx
100005842: 89 c8                       	movl	%ecx, %eax
100005844: c1 f8 1f                    	sarl	$31, %eax
100005847: c1 e8 12                    	shrl	$18, %eax
10000584a: 01 c8                       	addl	%ecx, %eax
10000584c: c1 f8 0e                    	sarl	$14, %eax
10000584f: 3d 80 00 00 00              	cmpl	$128, %eax
100005854: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x1ab>
100005856: b8 7f 00 00 00              	movl	$127, %eax
10000585b: 83 f8 81                    	cmpl	$-127, %eax
10000585e: 7f 80                       	jg	-128 <__ZN11LineNetwork7forwardEv+0x130>
100005860: b8 81 00 00 00              	movl	$129, %eax
100005865: e9 76 ff ff ff              	jmp	-138 <__ZN11LineNetwork7forwardEv+0x130>
10000586a: 4c 8b 7d 88                 	movq	-120(%rbp), %r15
10000586e: 4c 89 ff                    	movq	%r15, %rdi
100005871: e8 2a f3 ff ff              	callq	-3286 <__ZN14ModelInterface11swap_bufferEv>
100005876: 4c 89 ff                    	movq	%r15, %rdi
100005879: e8 72 f1 ff ff              	callq	-3726 <__ZN14ModelInterface13output_bufferEv>
10000587e: 49 89 c6                    	movq	%rax, %r14
100005881: 4c 89 ff                    	movq	%r15, %rdi
100005884: e8 57 f1 ff ff              	callq	-3753 <__ZN14ModelInterface12input_bufferEv>
100005889: 4c 89 f7                    	movq	%r14, %rdi
10000588c: 48 89 c6                    	movq	%rax, %rsi
10000588f: ba 00 00 02 00              	movl	$131072, %edx
100005894: e8 17 f3 ff ff              	callq	-3305 <__Z4ReLUPaS_j>
100005899: 4c 89 ff                    	movq	%r15, %rdi
10000589c: e8 ff f2 ff ff              	callq	-3329 <__ZN14ModelInterface11swap_bufferEv>
1000058a1: 4c 89 ff                    	movq	%r15, %rdi
1000058a4: e8 47 f1 ff ff              	callq	-3769 <__ZN14ModelInterface13output_bufferEv>
1000058a9: 49 89 c5                    	movq	%rax, %r13
1000058ac: 4c 89 ff                    	movq	%r15, %rdi
1000058af: e8 2c f1 ff ff              	callq	-3796 <__ZN14ModelInterface12input_bufferEv>
1000058b4: 48 89 45 c8                 	movq	%rax, -56(%rbp)
1000058b8: 31 c0                       	xorl	%eax, %eax
1000058ba: eb 1c                       	jmp	28 <__ZN11LineNetwork7forwardEv+0x228>
1000058bc: 0f 1f 40 00                 	nopl	(%rax)
1000058c0: 48 8b 45 c0                 	movq	-64(%rbp), %rax
1000058c4: 48 ff c0                    	incq	%rax
1000058c7: 4c 8b 6d b8                 	movq	-72(%rbp), %r13
1000058cb: 49 ff c5                    	incq	%r13
1000058ce: 48 83 f8 10                 	cmpq	$16, %rax
1000058d2: 0f 84 ff 00 00 00           	je	255 <__ZN11LineNetwork7forwardEv+0x327>
1000058d8: 48 89 45 c0                 	movq	%rax, -64(%rbp)
1000058dc: 48 8d 04 c5 00 00 00 00     	leaq	(,%rax,8), %rax
1000058e4: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
1000058e8: 48 8d 05 11 1d 00 00        	leaq	7441(%rip), %rax
1000058ef: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
1000058f3: 48 89 55 90                 	movq	%rdx, -112(%rbp)
1000058f7: 48 8d 54 08 18              	leaq	24(%rax,%rcx), %rdx
1000058fc: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005900: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
100005904: 48 8d 44 08 30              	leaq	48(%rax,%rcx), %rax
100005909: 48 89 45 a0                 	movq	%rax, -96(%rbp)
10000590d: 4c 89 6d b8                 	movq	%r13, -72(%rbp)
100005911: 4c 8b 7d c8                 	movq	-56(%rbp), %r15
100005915: 31 c0                       	xorl	%eax, %eax
100005917: eb 23                       	jmp	35 <__ZN11LineNetwork7forwardEv+0x28c>
100005919: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005920: 4c 8b 7d a8                 	movq	-88(%rbp), %r15
100005924: 49 81 c7 00 08 00 00        	addq	$2048, %r15
10000592b: 49 81 c5 00 04 00 00        	addq	$1024, %r13
100005932: 48 8b 45 b0                 	movq	-80(%rbp), %rax
100005936: 48 83 f8 7d                 	cmpq	$125, %rax
10000593a: 73 84                       	jae	-124 <__ZN11LineNetwork7forwardEv+0x210>
10000593c: 48 83 c0 02                 	addq	$2, %rax
100005940: 48 89 45 b0                 	movq	%rax, -80(%rbp)
100005944: 4c 89 7d a8                 	movq	%r15, -88(%rbp)
100005948: 31 db                       	xorl	%ebx, %ebx
10000594a: eb 17                       	jmp	23 <__ZN11LineNetwork7forwardEv+0x2b3>
10000594c: 0f 1f 40 00                 	nopl	(%rax)
100005950: 41 88 44 dd 00              	movb	%al, (%r13,%rbx,8)
100005955: 48 83 c3 02                 	addq	$2, %rbx
100005959: 49 83 c7 10                 	addq	$16, %r15
10000595d: 48 83 fb 7d                 	cmpq	$125, %rbx
100005961: 73 bd                       	jae	-67 <__ZN11LineNetwork7forwardEv+0x270>
100005963: 4c 89 ff                    	movq	%r15, %rdi
100005966: 48 8b 75 90                 	movq	-112(%rbp), %rsi
10000596a: e8 b1 10 00 00              	callq	4273 <__ZN11LineNetwork7forwardEv+0x1370>
10000596f: 41 89 c6                    	movl	%eax, %r14d
100005972: 49 8d bf 00 04 00 00        	leaq	1024(%r15), %rdi
100005979: 48 8b 75 98                 	movq	-104(%rbp), %rsi
10000597d: e8 9e 10 00 00              	callq	4254 <__ZN11LineNetwork7forwardEv+0x1370>
100005982: 41 89 c4                    	movl	%eax, %r12d
100005985: 45 01 f4                    	addl	%r14d, %r12d
100005988: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
10000598f: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005993: e8 88 10 00 00              	callq	4232 <__ZN11LineNetwork7forwardEv+0x1370>
100005998: 44 01 e0                    	addl	%r12d, %eax
10000599b: 48 8d 0d de 20 00 00        	leaq	8414(%rip), %rcx
1000059a2: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
1000059a6: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
1000059aa: 01 c1                       	addl	%eax, %ecx
1000059ac: 6b c9 39                    	imull	$57, %ecx, %ecx
1000059af: 89 c8                       	movl	%ecx, %eax
1000059b1: c1 f8 1f                    	sarl	$31, %eax
1000059b4: c1 e8 12                    	shrl	$18, %eax
1000059b7: 01 c8                       	addl	%ecx, %eax
1000059b9: c1 f8 0e                    	sarl	$14, %eax
1000059bc: 3d 80 00 00 00              	cmpl	$128, %eax
1000059c1: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x318>
1000059c3: b8 7f 00 00 00              	movl	$127, %eax
1000059c8: 83 f8 81                    	cmpl	$-127, %eax
1000059cb: 7f 83                       	jg	-125 <__ZN11LineNetwork7forwardEv+0x2a0>
1000059cd: b8 81 00 00 00              	movl	$129, %eax
1000059d2: e9 79 ff ff ff              	jmp	-135 <__ZN11LineNetwork7forwardEv+0x2a0>
1000059d7: 4c 8b 7d 88                 	movq	-120(%rbp), %r15
1000059db: 4c 89 ff                    	movq	%r15, %rdi
1000059de: e8 bd f1 ff ff              	callq	-3651 <__ZN14ModelInterface11swap_bufferEv>
1000059e3: 4c 89 ff                    	movq	%r15, %rdi
1000059e6: e8 05 f0 ff ff              	callq	-4091 <__ZN14ModelInterface13output_bufferEv>
1000059eb: 49 89 c6                    	movq	%rax, %r14
1000059ee: 4c 89 ff                    	movq	%r15, %rdi
1000059f1: e8 ea ef ff ff              	callq	-4118 <__ZN14ModelInterface12input_bufferEv>
1000059f6: 4c 89 f7                    	movq	%r14, %rdi
1000059f9: 48 89 c6                    	movq	%rax, %rsi
1000059fc: ba 00 00 01 00              	movl	$65536, %edx
100005a01: e8 aa f1 ff ff              	callq	-3670 <__Z4ReLUPaS_j>
100005a06: 4c 89 ff                    	movq	%r15, %rdi
100005a09: e8 92 f1 ff ff              	callq	-3694 <__ZN14ModelInterface11swap_bufferEv>
100005a0e: 4c 89 ff                    	movq	%r15, %rdi
100005a11: e8 da ef ff ff              	callq	-4134 <__ZN14ModelInterface13output_bufferEv>
100005a16: 48 89 c3                    	movq	%rax, %rbx
100005a19: 4c 89 ff                    	movq	%r15, %rdi
100005a1c: e8 bf ef ff ff              	callq	-4161 <__ZN14ModelInterface12input_bufferEv>
100005a21: 48 89 45 80                 	movq	%rax, -128(%rbp)
100005a25: 31 c0                       	xorl	%eax, %eax
100005a27: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x398>
100005a29: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005a30: 48 8b 45 c8                 	movq	-56(%rbp), %rax
100005a34: 48 ff c0                    	incq	%rax
100005a37: 48 8b 5d c0                 	movq	-64(%rbp), %rbx
100005a3b: 48 ff c3                    	incq	%rbx
100005a3e: 48 83 f8 20                 	cmpq	$32, %rax
100005a42: 0f 84 17 01 00 00           	je	279 <__ZN11LineNetwork7forwardEv+0x4af>
100005a48: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100005a4c: 48 c1 e0 04                 	shlq	$4, %rax
100005a50: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005a54: 48 8d 05 35 20 00 00        	leaq	8245(%rip), %rax
100005a5b: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005a5f: 48 89 55 90                 	movq	%rdx, -112(%rbp)
100005a63: 48 8d 54 08 30              	leaq	48(%rax,%rcx), %rdx
100005a68: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005a6c: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
100005a70: 48 8d 44 08 60              	leaq	96(%rax,%rcx), %rax
100005a75: 48 89 45 a0                 	movq	%rax, -96(%rbp)
100005a79: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
100005a7d: 4c 8b 7d 80                 	movq	-128(%rbp), %r15
100005a81: 31 c0                       	xorl	%eax, %eax
100005a83: eb 2b                       	jmp	43 <__ZN11LineNetwork7forwardEv+0x400>
100005a85: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005a8f: 90                          	nop
100005a90: 4c 8b 7d b0                 	movq	-80(%rbp), %r15
100005a94: 49 81 c7 00 08 00 00        	addq	$2048, %r15
100005a9b: 48 8b 5d a8                 	movq	-88(%rbp), %rbx
100005a9f: 48 81 c3 00 04 00 00        	addq	$1024, %rbx
100005aa6: 48 8b 45 b8                 	movq	-72(%rbp), %rax
100005aaa: 48 83 f8 3d                 	cmpq	$61, %rax
100005aae: 73 80                       	jae	-128 <__ZN11LineNetwork7forwardEv+0x380>
100005ab0: 48 83 c0 02                 	addq	$2, %rax
100005ab4: 48 89 45 b8                 	movq	%rax, -72(%rbp)
100005ab8: 48 89 5d a8                 	movq	%rbx, -88(%rbp)
100005abc: 4c 89 7d b0                 	movq	%r15, -80(%rbp)
100005ac0: 45 31 f6                    	xorl	%r14d, %r14d
100005ac3: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x434>
100005ac5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005acf: 90                          	nop
100005ad0: 88 03                       	movb	%al, (%rbx)
100005ad2: 49 83 c6 02                 	addq	$2, %r14
100005ad6: 49 83 c7 20                 	addq	$32, %r15
100005ada: 48 83 c3 20                 	addq	$32, %rbx
100005ade: 49 83 fe 3d                 	cmpq	$61, %r14
100005ae2: 73 ac                       	jae	-84 <__ZN11LineNetwork7forwardEv+0x3e0>
100005ae4: 4c 89 ff                    	movq	%r15, %rdi
100005ae7: 48 8b 75 90                 	movq	-112(%rbp), %rsi
100005aeb: e8 f0 0f 00 00              	callq	4080 <__ZN11LineNetwork7forwardEv+0x1430>
100005af0: 41 89 c4                    	movl	%eax, %r12d
100005af3: 49 8d bf 00 04 00 00        	leaq	1024(%r15), %rdi
100005afa: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100005afe: e8 dd 0f 00 00              	callq	4061 <__ZN11LineNetwork7forwardEv+0x1430>
100005b03: 41 89 c5                    	movl	%eax, %r13d
100005b06: 45 01 e5                    	addl	%r12d, %r13d
100005b09: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
100005b10: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005b14: e8 c7 0f 00 00              	callq	4039 <__ZN11LineNetwork7forwardEv+0x1430>
100005b19: 44 01 e8                    	addl	%r13d, %eax
100005b1c: 48 8d 0d 6d 31 00 00        	leaq	12653(%rip), %rcx
100005b23: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
100005b27: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
100005b2b: 01 c1                       	addl	%eax, %ecx
100005b2d: c1 e1 04                    	shll	$4, %ecx
100005b30: 8d 0c 49                    	leal	(%rcx,%rcx,2), %ecx
100005b33: 89 c8                       	movl	%ecx, %eax
100005b35: c1 f8 1f                    	sarl	$31, %eax
100005b38: c1 e8 12                    	shrl	$18, %eax
100005b3b: 01 c8                       	addl	%ecx, %eax
100005b3d: c1 f8 0e                    	sarl	$14, %eax
100005b40: 3d 80 00 00 00              	cmpl	$128, %eax
100005b45: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x49c>
100005b47: b8 7f 00 00 00              	movl	$127, %eax
100005b4c: 83 f8 81                    	cmpl	$-127, %eax
100005b4f: 0f 8f 7b ff ff ff           	jg	-133 <__ZN11LineNetwork7forwardEv+0x420>
100005b55: b8 81 00 00 00              	movl	$129, %eax
100005b5a: e9 71 ff ff ff              	jmp	-143 <__ZN11LineNetwork7forwardEv+0x420>
100005b5f: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100005b63: 48 89 df                    	movq	%rbx, %rdi
100005b66: e8 35 f0 ff ff              	callq	-4043 <__ZN14ModelInterface11swap_bufferEv>
100005b6b: 48 89 df                    	movq	%rbx, %rdi
100005b6e: e8 7d ee ff ff              	callq	-4483 <__ZN14ModelInterface13output_bufferEv>
100005b73: 49 89 c6                    	movq	%rax, %r14
100005b76: 48 89 df                    	movq	%rbx, %rdi
100005b79: e8 62 ee ff ff              	callq	-4510 <__ZN14ModelInterface12input_bufferEv>
100005b7e: 4c 89 f7                    	movq	%r14, %rdi
100005b81: 48 89 c6                    	movq	%rax, %rsi
100005b84: ba 00 80 00 00              	movl	$32768, %edx
100005b89: e8 22 f0 ff ff              	callq	-4062 <__Z4ReLUPaS_j>
100005b8e: 48 89 df                    	movq	%rbx, %rdi
100005b91: e8 0a f0 ff ff              	callq	-4086 <__ZN14ModelInterface11swap_bufferEv>
100005b96: 48 89 df                    	movq	%rbx, %rdi
100005b99: e8 52 ee ff ff              	callq	-4526 <__ZN14ModelInterface13output_bufferEv>
100005b9e: 49 89 c4                    	movq	%rax, %r12
100005ba1: 48 89 df                    	movq	%rbx, %rdi
100005ba4: e8 37 ee ff ff              	callq	-4553 <__ZN14ModelInterface12input_bufferEv>
100005ba9: 49 89 c6                    	movq	%rax, %r14
100005bac: 31 c0                       	xorl	%eax, %eax
100005bae: 4c 8d 3d fb 30 00 00        	leaq	12539(%rip), %r15
100005bb5: eb 21                       	jmp	33 <__ZN11LineNetwork7forwardEv+0x528>
100005bb7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100005bc0: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100005bc4: 48 ff c0                    	incq	%rax
100005bc7: 49 83 c4 20                 	addq	$32, %r12
100005bcb: 49 81 c6 00 04 00 00        	addq	$1024, %r14
100005bd2: 48 83 f8 20                 	cmpq	$32, %rax
100005bd6: 74 63                       	je	99 <__ZN11LineNetwork7forwardEv+0x58b>
100005bd8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100005bdc: 4c 89 f3                    	movq	%r14, %rbx
100005bdf: 45 31 ed                    	xorl	%r13d, %r13d
100005be2: eb 1d                       	jmp	29 <__ZN11LineNetwork7forwardEv+0x551>
100005be4: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005bee: 66 90                       	nop
100005bf0: 43 88 04 2c                 	movb	%al, (%r12,%r13)
100005bf4: 49 ff c5                    	incq	%r13
100005bf7: 48 83 c3 20                 	addq	$32, %rbx
100005bfb: 49 83 fd 20                 	cmpq	$32, %r13
100005bff: 74 bf                       	je	-65 <__ZN11LineNetwork7forwardEv+0x510>
100005c01: 48 89 df                    	movq	%rbx, %rdi
100005c04: 4c 89 fe                    	movq	%r15, %rsi
100005c07: e8 54 11 00 00              	callq	4436 <__ZN11LineNetwork7forwardEv+0x16b0>
100005c0c: c1 e0 05                    	shll	$5, %eax
100005c0f: 89 c1                       	movl	%eax, %ecx
100005c11: 83 c1 20                    	addl	$32, %ecx
100005c14: c1 f9 1f                    	sarl	$31, %ecx
100005c17: c1 e9 12                    	shrl	$18, %ecx
100005c1a: 8d 04 08                    	leal	(%rax,%rcx), %eax
100005c1d: 83 c0 20                    	addl	$32, %eax
100005c20: c1 f8 0e                    	sarl	$14, %eax
100005c23: 3d 80 00 00 00              	cmpl	$128, %eax
100005c28: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x57f>
100005c2a: b8 7f 00 00 00              	movl	$127, %eax
100005c2f: 83 f8 81                    	cmpl	$-127, %eax
100005c32: 7f bc                       	jg	-68 <__ZN11LineNetwork7forwardEv+0x540>
100005c34: b8 81 00 00 00              	movl	$129, %eax
100005c39: eb b5                       	jmp	-75 <__ZN11LineNetwork7forwardEv+0x540>
100005c3b: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100005c3f: 48 89 df                    	movq	%rbx, %rdi
100005c42: e8 59 ef ff ff              	callq	-4263 <__ZN14ModelInterface11swap_bufferEv>
100005c47: 48 89 df                    	movq	%rbx, %rdi
100005c4a: 48 83 c4 58                 	addq	$88, %rsp
100005c4e: 5b                          	popq	%rbx
100005c4f: 41 5c                       	popq	%r12
100005c51: 41 5d                       	popq	%r13
100005c53: 41 5e                       	popq	%r14
100005c55: 41 5f                       	popq	%r15
100005c57: 5d                          	popq	%rbp
100005c58: e9 43 ef ff ff              	jmp	-4285 <__ZN14ModelInterface11swap_bufferEv>
100005c5d: 0f 1f 00                    	nopl	(%rax)
100005c60: 55                          	pushq	%rbp
100005c61: 48 89 e5                    	movq	%rsp, %rbp
100005c64: 41 57                       	pushq	%r15
100005c66: 41 56                       	pushq	%r14
100005c68: 41 55                       	pushq	%r13
100005c6a: 41 54                       	pushq	%r12
100005c6c: 53                          	pushq	%rbx
100005c6d: 48 83 e4 e0                 	andq	$-32, %rsp
100005c71: 48 81 ec e0 02 00 00        	subq	$736, %rsp
100005c78: 48 89 4c 24 50              	movq	%rcx, 80(%rsp)
100005c7d: 48 89 54 24 48              	movq	%rdx, 72(%rsp)
100005c82: 49 89 ff                    	movq	%rdi, %r15
100005c85: c4 c1 79 6e c0              	vmovd	%r8d, %xmm0
100005c8a: c4 e2 7d 58 c8              	vpbroadcastd	%xmm0, %ymm1
100005c8f: 48 8d 86 01 04 00 00        	leaq	1025(%rsi), %rax
100005c96: 48 89 44 24 40              	movq	%rax, 64(%rsp)
100005c9b: 48 8d 86 02 04 00 00        	leaq	1026(%rsi), %rax
100005ca2: 48 89 44 24 38              	movq	%rax, 56(%rsp)
100005ca7: 45 31 c9                    	xorl	%r9d, %r9d
100005caa: c5 fd 6f 15 2e 16 00 00     	vmovdqa	5678(%rip), %ymm2
100005cb2: 44 89 44 24 14              	movl	%r8d, 20(%rsp)
100005cb7: 48 89 74 24 58              	movq	%rsi, 88(%rsp)
100005cbc: c5 fd 7f 8c 24 60 02 00 00  	vmovdqa	%ymm1, 608(%rsp)
100005cc5: eb 19                       	jmp	25 <__ZN11LineNetwork7forwardEv+0x630>
100005cc7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100005cd0: 49 ff c1                    	incq	%r9
100005cd3: 48 ff c7                    	incq	%rdi
100005cd6: 49 83 f9 08                 	cmpq	$8, %r9
100005cda: 0f 84 f2 0c 00 00           	je	3314 <__ZN11LineNetwork7forwardEv+0x1322>
100005ce0: 49 8d 81 f1 07 00 00        	leaq	2033(%r9), %rax
100005ce7: 48 89 84 24 88 00 00 00     	movq	%rax, 136(%rsp)
100005cef: 4b 8d 04 c9                 	leaq	(%r9,%r9,8), %rax
100005cf3: 48 8b 54 24 48              	movq	72(%rsp), %rdx
100005cf8: 48 8d 0c 02                 	leaq	(%rdx,%rax), %rcx
100005cfc: 48 83 c1 09                 	addq	$9, %rcx
100005d00: 48 89 8c 24 80 00 00 00     	movq	%rcx, 128(%rsp)
100005d08: 48 8b 4c 24 50              	movq	80(%rsp), %rcx
100005d0d: 48 8d 5c 01 01              	leaq	1(%rcx,%rax), %rbx
100005d12: 48 89 5c 24 78              	movq	%rbx, 120(%rsp)
100005d17: 4c 8d 14 02                 	leaq	(%rdx,%rax), %r10
100005d1b: 4c 8d 1c 01                 	leaq	(%rcx,%rax), %r11
100005d1f: 48 8d 44 02 08              	leaq	8(%rdx,%rax), %rax
100005d24: 48 89 44 24 70              	movq	%rax, 112(%rsp)
100005d29: c4 c1 f9 6e c1              	vmovq	%r9, %xmm0
100005d2e: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005d33: 41 be 00 00 00 00           	movl	$0, %r14d
100005d39: 48 8b 44 24 38              	movq	56(%rsp), %rax
100005d3e: 48 89 44 24 30              	movq	%rax, 48(%rsp)
100005d43: 48 8b 44 24 40              	movq	64(%rsp), %rax
100005d48: 31 c9                       	xorl	%ecx, %ecx
100005d4a: 31 d2                       	xorl	%edx, %edx
100005d4c: 48 89 54 24 08              	movq	%rdx, 8(%rsp)
100005d51: 4c 89 4c 24 68              	movq	%r9, 104(%rsp)
100005d56: 48 89 7c 24 60              	movq	%rdi, 96(%rsp)
100005d5b: 4c 89 54 24 20              	movq	%r10, 32(%rsp)
100005d60: 4c 89 5c 24 18              	movq	%r11, 24(%rsp)
100005d65: c5 fd 7f 84 24 80 02 00 00  	vmovdqa	%ymm0, 640(%rsp)
100005d6e: eb 38                       	jmp	56 <__ZN11LineNetwork7forwardEv+0x6f8>
100005d70: 48 8b 8c 24 90 00 00 00     	movq	144(%rsp), %rcx
100005d78: 48 ff c1                    	incq	%rcx
100005d7b: 48 8b 44 24 28              	movq	40(%rsp), %rax
100005d80: 48 05 00 04 00 00           	addq	$1024, %rax
100005d86: 48 81 44 24 30 00 04 00 00  	addq	$1024, 48(%rsp)
100005d8f: 49 81 c6 00 01 00 00        	addq	$256, %r14
100005d96: 48 81 7c 24 08 fd 01 00 00  	cmpq	$509, 8(%rsp)
100005d9f: 4d 89 e9                    	movq	%r13, %r9
100005da2: 0f 83 28 ff ff ff           	jae	-216 <__ZN11LineNetwork7forwardEv+0x620>
100005da8: 48 89 44 24 28              	movq	%rax, 40(%rsp)
100005dad: 4c 89 b4 24 98 00 00 00     	movq	%r14, 152(%rsp)
100005db5: 48 89 cb                    	movq	%rcx, %rbx
100005db8: 48 c1 e3 0b                 	shlq	$11, %rbx
100005dbc: 4d 89 cd                    	movq	%r9, %r13
100005dbf: 49 8d 04 19                 	leaq	(%r9,%rbx), %rax
100005dc3: 4c 01 f8                    	addq	%r15, %rax
100005dc6: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100005dce: 4c 01 fb                    	addq	%r15, %rbx
100005dd1: 48 89 ca                    	movq	%rcx, %rdx
100005dd4: 48 c1 e2 0a                 	shlq	$10, %rdx
100005dd8: 4c 8d 0c 16                 	leaq	(%rsi,%rdx), %r9
100005ddc: 49 81 c1 ff 05 00 00        	addq	$1535, %r9
100005de3: 48 01 f2                    	addq	%rsi, %rdx
100005de6: 4c 39 c8                    	cmpq	%r9, %rax
100005de9: 41 0f 92 c4                 	setb	%r12b
100005ded: 48 39 da                    	cmpq	%rbx, %rdx
100005df0: 41 0f 92 c2                 	setb	%r10b
100005df4: 48 3b 84 24 80 00 00 00     	cmpq	128(%rsp), %rax
100005dfc: 41 0f 92 c6                 	setb	%r14b
100005e00: 48 39 5c 24 70              	cmpq	%rbx, 112(%rsp)
100005e05: 4c 89 da                    	movq	%r11, %rdx
100005e08: 41 0f 92 c3                 	setb	%r11b
100005e0c: 48 3b 44 24 78              	cmpq	120(%rsp), %rax
100005e11: 0f 92 c0                    	setb	%al
100005e14: 48 39 da                    	cmpq	%rbx, %rdx
100005e17: 41 0f 92 c1                 	setb	%r9b
100005e1b: 45 84 d4                    	testb	%r10b, %r12b
100005e1e: 48 89 8c 24 90 00 00 00     	movq	%rcx, 144(%rsp)
100005e26: 0f 85 84 0a 00 00           	jne	2692 <__ZN11LineNetwork7forwardEv+0x1200>
100005e2c: 45 20 de                    	andb	%r11b, %r14b
100005e2f: 0f 85 7b 0a 00 00           	jne	2683 <__ZN11LineNetwork7forwardEv+0x1200>
100005e35: ba 00 00 00 00              	movl	$0, %edx
100005e3a: 44 20 c8                    	andb	%r9b, %al
100005e3d: 0f 85 6f 0a 00 00           	jne	2671 <__ZN11LineNetwork7forwardEv+0x1202>
100005e43: 48 8b 44 24 08              	movq	8(%rsp), %rax
100005e48: 48 c1 e0 07                 	shlq	$7, %rax
100005e4c: c4 e1 f9 6e c0              	vmovq	%rax, %xmm0
100005e51: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005e56: c5 fd 7f 84 24 a0 02 00 00  	vmovdqa	%ymm0, 672(%rsp)
100005e5f: 45 31 db                    	xorl	%r11d, %r11d
100005e62: c5 fc 28 05 56 14 00 00     	vmovaps	5206(%rip), %ymm0
100005e6a: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100005e73: c5 fc 28 05 25 14 00 00     	vmovaps	5157(%rip), %ymm0
100005e7b: c5 fc 29 84 24 20 02 00 00  	vmovaps	%ymm0, 544(%rsp)
100005e84: c5 fc 28 05 f4 13 00 00     	vmovaps	5108(%rip), %ymm0
100005e8c: c5 fc 29 84 24 00 02 00 00  	vmovaps	%ymm0, 512(%rsp)
100005e95: c5 fc 28 05 c3 13 00 00     	vmovaps	5059(%rip), %ymm0
100005e9d: c5 fc 29 84 24 e0 01 00 00  	vmovaps	%ymm0, 480(%rsp)
100005ea6: c5 fc 28 05 92 13 00 00     	vmovaps	5010(%rip), %ymm0
100005eae: c5 fc 29 84 24 c0 01 00 00  	vmovaps	%ymm0, 448(%rsp)
100005eb7: c5 fc 28 05 61 13 00 00     	vmovaps	4961(%rip), %ymm0
100005ebf: c5 fc 29 84 24 a0 01 00 00  	vmovaps	%ymm0, 416(%rsp)
100005ec8: c5 fc 28 05 30 13 00 00     	vmovaps	4912(%rip), %ymm0
100005ed0: c5 fc 29 84 24 80 01 00 00  	vmovaps	%ymm0, 384(%rsp)
100005ed9: c5 fc 28 05 ff 12 00 00     	vmovaps	4863(%rip), %ymm0
100005ee1: c5 fc 29 84 24 60 01 00 00  	vmovaps	%ymm0, 352(%rsp)
100005eea: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100005ef0: 48 8b 4c 24 28              	movq	40(%rsp), %rcx
100005ef5: c4 a1 7e 6f 84 59 1f fc ff ff       	vmovdqu	-993(%rcx,%r11,2), %ymm0
100005eff: c4 e2 7d 00 c2              	vpshufb	%ymm2, %ymm0, %ymm0
100005f04: c4 a1 7e 6f 8c 59 ff fb ff ff       	vmovdqu	-1025(%rcx,%r11,2), %ymm1
100005f0e: c4 21 7e 6f 84 59 00 fc ff ff       	vmovdqu	-1024(%rcx,%r11,2), %ymm8
100005f18: c5 7d 6f 1d e0 13 00 00     	vmovdqa	5088(%rip), %ymm11
100005f20: c4 c2 75 00 cb              	vpshufb	%ymm11, %ymm1, %ymm1
100005f25: c4 e3 75 02 c0 cc           	vpblendd	$204, %ymm0, %ymm1, %ymm0
100005f2b: c4 e3 fd 00 c8 d8           	vpermq	$216, %ymm0, %ymm1
100005f31: c4 a1 7a 6f 94 59 0f fc ff ff       	vmovdqu	-1009(%rcx,%r11,2), %xmm2
100005f3b: c5 f9 6f 1d 6d 12 00 00     	vmovdqa	4717(%rip), %xmm3
100005f43: c4 e2 69 00 d3              	vpshufb	%xmm3, %xmm2, %xmm2
100005f48: c5 79 6f e3                 	vmovdqa	%xmm3, %xmm12
100005f4c: c4 62 7d 21 ca              	vpmovsxbd	%xmm2, %ymm9
100005f51: c4 63 fd 00 d0 db           	vpermq	$219, %ymm0, %ymm10
100005f57: 48 8b 44 24 20              	movq	32(%rsp), %rax
100005f5c: c4 e2 79 78 00              	vpbroadcastb	(%rax), %xmm0
100005f61: c4 e2 7d 21 d0              	vpmovsxbd	%xmm0, %ymm2
100005f66: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
100005f6b: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100005f74: c4 62 7d 21 c9              	vpmovsxbd	%xmm1, %ymm9
100005f79: c4 42 7d 21 d2              	vpmovsxbd	%xmm10, %ymm10
100005f7e: c4 21 7e 6f ac 59 20 fc ff ff       	vmovdqu	-992(%rcx,%r11,2), %ymm13
100005f88: c4 62 15 00 3d 4f 13 00 00  	vpshufb	4943(%rip), %ymm13, %ymm15
100005f91: c4 c2 3d 00 fb              	vpshufb	%ymm11, %ymm8, %ymm7
100005f96: c4 c3 45 02 ff cc           	vpblendd	$204, %ymm15, %ymm7, %ymm7
100005f9c: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
100005fa2: c4 63 fd 00 ff d8           	vpermq	$216, %ymm7, %ymm15
100005fa8: c5 fd 6f 05 70 13 00 00     	vmovdqa	4976(%rip), %ymm0
100005fb0: c4 62 15 00 e8              	vpshufb	%ymm0, %ymm13, %ymm13
100005fb5: c5 fd 6f 05 83 13 00 00     	vmovdqa	4995(%rip), %ymm0
100005fbd: c4 62 3d 00 c0              	vpshufb	%ymm0, %ymm8, %ymm8
100005fc2: c4 c3 3d 02 f5 cc           	vpblendd	$204, %ymm13, %ymm8, %ymm6
100005fc8: c4 e3 fd 00 ee d8           	vpermq	$216, %ymm6, %ymm5
100005fce: c4 e2 7d 21 e1              	vpmovsxbd	%xmm1, %ymm4
100005fd3: c4 c2 7d 21 df              	vpmovsxbd	%xmm15, %ymm3
100005fd8: c4 e3 fd 00 cf db           	vpermq	$219, %ymm7, %ymm1
100005fde: c4 62 7d 21 e9              	vpmovsxbd	%xmm1, %ymm13
100005fe3: c4 43 7d 39 ff 01           	vextracti128	$1, %ymm15, %xmm15
100005fe9: c4 42 6d 40 c2              	vpmulld	%ymm10, %ymm2, %ymm8
100005fee: c4 21 7a 6f 94 59 10 fc ff ff       	vmovdqu	-1008(%rcx,%r11,2), %xmm10
100005ff8: c4 c2 29 00 fc              	vpshufb	%xmm12, %xmm10, %xmm7
100005ffd: c4 62 79 78 70 01           	vpbroadcastb	1(%rax), %xmm14
100006003: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
100006008: c5 fd 7f 84 24 a0 00 00 00  	vmovdqa	%ymm0, 160(%rsp)
100006011: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
100006016: c4 42 7d 21 f6              	vpmovsxbd	%xmm14, %ymm14
10000601b: c4 e2 0d 40 ff              	vpmulld	%ymm7, %ymm14, %ymm7
100006020: c4 42 0d 40 ed              	vpmulld	%ymm13, %ymm14, %ymm13
100006025: c4 42 7d 21 e7              	vpmovsxbd	%xmm15, %ymm12
10000602a: c4 c3 7d 39 ef 01           	vextracti128	$1, %ymm5, %xmm15
100006030: c4 e3 fd 00 f6 db           	vpermq	$219, %ymm6, %ymm6
100006036: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
10000603b: c4 62 0d 40 cb              	vpmulld	%ymm3, %ymm14, %ymm9
100006040: c4 e2 7d 21 dd              	vpmovsxbd	%xmm5, %ymm3
100006045: c5 f9 6f 05 73 11 00 00     	vmovdqa	4467(%rip), %xmm0
10000604d: c4 e2 29 00 e8              	vpshufb	%xmm0, %xmm10, %xmm5
100006052: c4 e2 79 78 40 02           	vpbroadcastb	2(%rax), %xmm0
100006058: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
10000605d: c4 c2 7d 21 cf              	vpmovsxbd	%xmm15, %ymm1
100006062: c4 62 7d 40 fb              	vpmulld	%ymm3, %ymm0, %ymm15
100006067: c4 62 7d 40 d6              	vpmulld	%ymm6, %ymm0, %ymm10
10000606c: c4 e2 6d 40 d4              	vpmulld	%ymm4, %ymm2, %ymm2
100006071: c5 fd 7f 94 24 40 01 00 00  	vmovdqa	%ymm2, 320(%rsp)
10000607a: c4 e2 7d 21 d5              	vpmovsxbd	%xmm5, %ymm2
10000607f: c4 e2 7d 40 d2              	vpmulld	%ymm2, %ymm0, %ymm2
100006084: c4 a1 7e 6f 9c 59 ff fd ff ff       	vmovdqu	-513(%rcx,%r11,2), %ymm3
10000608e: c4 c2 0d 40 e4              	vpmulld	%ymm12, %ymm14, %ymm4
100006093: c5 fd 7f a4 24 00 01 00 00  	vmovdqa	%ymm4, 256(%rsp)
10000609c: c4 a1 7e 6f a4 59 1f fe ff ff       	vmovdqu	-481(%rcx,%r11,2), %ymm4
1000060a6: c4 e2 5d 00 25 31 12 00 00  	vpshufb	4657(%rip), %ymm4, %ymm4
1000060af: c4 c2 65 00 db              	vpshufb	%ymm11, %ymm3, %ymm3
1000060b4: c4 e3 65 02 dc cc           	vpblendd	$204, %ymm4, %ymm3, %ymm3
1000060ba: c4 e2 7d 40 c1              	vpmulld	%ymm1, %ymm0, %ymm0
1000060bf: c5 fd 7f 84 24 20 01 00 00  	vmovdqa	%ymm0, 288(%rsp)
1000060c8: c4 e3 fd 00 c3 d8           	vpermq	$216, %ymm3, %ymm0
1000060ce: c4 e2 7d 21 c8              	vpmovsxbd	%xmm0, %ymm1
1000060d3: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
1000060d9: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000060de: c4 e3 7d 39 c0 01           	vextracti128	$1, %ymm0, %xmm0
1000060e4: c5 c5 fe a4 24 c0 00 00 00  	vpaddd	192(%rsp), %ymm7, %ymm4
1000060ed: c4 a1 7a 6f ac 59 0f fe ff ff       	vmovdqu	-497(%rcx,%r11,2), %xmm5
1000060f7: c5 79 6f 35 b1 10 00 00     	vmovdqa	4273(%rip), %xmm14
1000060ff: c4 c2 51 00 ee              	vpshufb	%xmm14, %xmm5, %xmm5
100006104: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006109: c4 e2 79 78 70 03           	vpbroadcastb	3(%rax), %xmm6
10000610f: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006114: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
100006119: c4 e2 4d 40 c0              	vpmulld	%ymm0, %ymm6, %ymm0
10000611e: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100006127: c4 e2 4d 40 db              	vpmulld	%ymm3, %ymm6, %ymm3
10000612c: c4 41 3d fe ed              	vpaddd	%ymm13, %ymm8, %ymm13
100006131: c4 e2 4d 40 c9              	vpmulld	%ymm1, %ymm6, %ymm1
100006136: c4 e2 4d 40 c5              	vpmulld	%ymm5, %ymm6, %ymm0
10000613b: c5 ed fe c0                 	vpaddd	%ymm0, %ymm2, %ymm0
10000613f: c5 35 fe 84 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm9, %ymm8
100006148: c5 dd fe c0                 	vpaddd	%ymm0, %ymm4, %ymm0
10000614c: c5 fd 7f 84 24 e0 00 00 00  	vmovdqa	%ymm0, 224(%rsp)
100006155: c4 a1 7e 6f a4 59 00 fe ff ff       	vmovdqu	-512(%rcx,%r11,2), %ymm4
10000615f: c4 a1 7e 6f ac 59 20 fe ff ff       	vmovdqu	-480(%rcx,%r11,2), %ymm5
100006169: c4 e2 55 00 35 6e 11 00 00  	vpshufb	4462(%rip), %ymm5, %ymm6
100006172: c4 c2 5d 00 fb              	vpshufb	%ymm11, %ymm4, %ymm7
100006177: c5 2d fe d3                 	vpaddd	%ymm3, %ymm10, %ymm10
10000617b: c4 e3 45 02 de cc           	vpblendd	$204, %ymm6, %ymm7, %ymm3
100006181: c4 e3 fd 00 f3 d8           	vpermq	$216, %ymm3, %ymm6
100006187: c4 e3 7d 39 f7 01           	vextracti128	$1, %ymm6, %xmm7
10000618d: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
100006192: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
100006198: c5 05 fe c9                 	vpaddd	%ymm1, %ymm15, %ymm9
10000619c: c4 e2 7d 21 cb              	vpmovsxbd	%xmm3, %ymm1
1000061a1: c4 e2 7d 21 de              	vpmovsxbd	%xmm6, %ymm3
1000061a6: c4 a1 7a 6f b4 59 10 fe ff ff       	vmovdqu	-496(%rcx,%r11,2), %xmm6
1000061b0: c4 e2 79 78 40 04           	vpbroadcastb	4(%rax), %xmm0
1000061b6: c4 c2 49 00 d6              	vpshufb	%xmm14, %xmm6, %xmm2
1000061bb: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000061c0: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
1000061c5: c4 e2 7d 40 db              	vpmulld	%ymm3, %ymm0, %ymm3
1000061ca: c4 62 7d 40 e1              	vpmulld	%ymm1, %ymm0, %ymm12
1000061cf: c4 e2 7d 40 cf              	vpmulld	%ymm7, %ymm0, %ymm1
1000061d4: c5 fd 7f 8c 24 a0 00 00 00  	vmovdqa	%ymm1, 160(%rsp)
1000061dd: c4 e2 7d 40 c2              	vpmulld	%ymm2, %ymm0, %ymm0
1000061e2: c4 e2 55 00 15 35 11 00 00  	vpshufb	4405(%rip), %ymm5, %ymm2
1000061eb: c4 e2 5d 00 25 4c 11 00 00  	vpshufb	4428(%rip), %ymm4, %ymm4
1000061f4: c4 e3 5d 02 d2 cc           	vpblendd	$204, %ymm2, %ymm4, %ymm2
1000061fa: c4 e2 79 78 60 05           	vpbroadcastb	5(%rax), %xmm4
100006200: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100006205: c4 e3 fd 00 ea db           	vpermq	$219, %ymm2, %ymm5
10000620b: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006210: c4 e2 5d 40 ed              	vpmulld	%ymm5, %ymm4, %ymm5
100006215: c5 9d fe ed                 	vpaddd	%ymm5, %ymm12, %ymm5
100006219: c4 e3 fd 00 d2 d8           	vpermq	$216, %ymm2, %ymm2
10000621f: c4 e2 7d 21 fa              	vpmovsxbd	%xmm2, %ymm7
100006224: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000622a: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000622f: c4 e2 49 00 35 88 0f 00 00  	vpshufb	3976(%rip), %xmm6, %xmm6
100006238: c4 e2 5d 40 ff              	vpmulld	%ymm7, %ymm4, %ymm7
10000623d: c4 62 5d 40 fa              	vpmulld	%ymm2, %ymm4, %ymm15
100006242: c4 e2 7d 21 d6              	vpmovsxbd	%xmm6, %ymm2
100006247: c4 e2 5d 40 d2              	vpmulld	%ymm2, %ymm4, %ymm2
10000624c: c5 fd fe c2                 	vpaddd	%ymm2, %ymm0, %ymm0
100006250: c4 a1 7e 6f 54 59 ff        	vmovdqu	-1(%rcx,%r11,2), %ymm2
100006257: c5 e5 fe df                 	vpaddd	%ymm7, %ymm3, %ymm3
10000625b: c4 a1 7e 6f 64 59 1f        	vmovdqu	31(%rcx,%r11,2), %ymm4
100006262: c4 e2 5d 00 25 75 10 00 00  	vpshufb	4213(%rip), %ymm4, %ymm4
10000626b: c4 c2 6d 00 d3              	vpshufb	%ymm11, %ymm2, %ymm2
100006270: c4 e3 6d 02 d4 cc           	vpblendd	$204, %ymm4, %ymm2, %ymm2
100006276: c4 e3 fd 00 e2 d8           	vpermq	$216, %ymm2, %ymm4
10000627c: c4 c1 15 fe f2              	vpaddd	%ymm10, %ymm13, %ymm6
100006281: c4 e3 7d 39 e7 01           	vextracti128	$1, %ymm4, %xmm7
100006287: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
10000628c: c4 e3 fd 00 d2 db           	vpermq	$219, %ymm2, %ymm2
100006292: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
100006297: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
10000629c: c4 41 3d fe c1              	vpaddd	%ymm9, %ymm8, %ymm8
1000062a1: c4 e2 79 78 48 06           	vpbroadcastb	6(%rax), %xmm1
1000062a7: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000062ac: c4 e2 75 40 e4              	vpmulld	%ymm4, %ymm1, %ymm4
1000062b1: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
1000062b5: c4 a1 7a 6f 64 59 0f        	vmovdqu	15(%rcx,%r11,2), %xmm4
1000062bc: c4 c2 59 00 e6              	vpshufb	%xmm14, %xmm4, %xmm4
1000062c1: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000062c6: c4 e2 75 40 d2              	vpmulld	%ymm2, %ymm1, %ymm2
1000062cb: c5 d5 fe d2                 	vpaddd	%ymm2, %ymm5, %ymm2
1000062cf: c4 62 75 40 ef              	vpmulld	%ymm7, %ymm1, %ymm13
1000062d4: c4 e2 75 40 cc              	vpmulld	%ymm4, %ymm1, %ymm1
1000062d9: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
1000062dd: c5 3d fe c3                 	vpaddd	%ymm3, %ymm8, %ymm8
1000062e1: c5 7d fe 94 24 e0 00 00 00  	vpaddd	224(%rsp), %ymm0, %ymm10
1000062ea: c4 a1 7e 6f 0c 59           	vmovdqu	(%rcx,%r11,2), %ymm1
1000062f0: c4 a1 7e 6f 5c 59 20        	vmovdqu	32(%rcx,%r11,2), %ymm3
1000062f7: c4 e2 65 00 25 e0 0f 00 00  	vpshufb	4064(%rip), %ymm3, %ymm4
100006300: c4 c2 75 00 eb              	vpshufb	%ymm11, %ymm1, %ymm5
100006305: c5 4d fe da                 	vpaddd	%ymm2, %ymm6, %ymm11
100006309: c4 e3 55 02 e4 cc           	vpblendd	$204, %ymm4, %ymm5, %ymm4
10000630f: c4 e3 fd 00 ec d8           	vpermq	$216, %ymm4, %ymm5
100006315: c4 e2 65 00 1d 02 10 00 00  	vpshufb	4098(%rip), %ymm3, %ymm3
10000631e: c4 e2 75 00 0d 19 10 00 00  	vpshufb	4121(%rip), %ymm1, %ymm1
100006327: c4 e3 75 02 cb cc           	vpblendd	$204, %ymm3, %ymm1, %ymm1
10000632d: c5 fd 6f 84 24 00 01 00 00  	vmovdqa	256(%rsp), %ymm0
100006336: c5 7d fe a4 24 40 01 00 00  	vpaddd	320(%rsp), %ymm0, %ymm12
10000633f: c4 e3 fd 00 f1 d8           	vpermq	$216, %ymm1, %ymm6
100006345: c4 e2 7d 21 fd              	vpmovsxbd	%xmm5, %ymm7
10000634a: c4 e3 fd 00 e4 db           	vpermq	$219, %ymm4, %ymm4
100006350: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100006355: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
10000635b: c5 fd 6f 84 24 c0 00 00 00  	vmovdqa	192(%rsp), %ymm0
100006364: c5 7d fe 8c 24 20 01 00 00  	vpaddd	288(%rsp), %ymm0, %ymm9
10000636d: c4 a1 7a 6f 44 59 10        	vmovdqu	16(%rcx,%r11,2), %xmm0
100006374: c4 c2 79 00 d6              	vpshufb	%xmm14, %xmm0, %xmm2
100006379: c4 e2 79 78 58 07           	vpbroadcastb	7(%rax), %xmm3
10000637f: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
100006384: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006389: c4 e2 65 40 e4              	vpmulld	%ymm4, %ymm3, %ymm4
10000638e: c4 e2 65 40 ff              	vpmulld	%ymm7, %ymm3, %ymm7
100006393: c4 e2 65 40 ed              	vpmulld	%ymm5, %ymm3, %ymm5
100006398: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000639d: c4 e2 65 40 d2              	vpmulld	%ymm2, %ymm3, %ymm2
1000063a2: c4 e2 79 78 58 08           	vpbroadcastb	8(%rax), %xmm3
1000063a8: c5 05 fe b4 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm15, %ymm14
1000063b1: c4 62 7d 21 fe              	vpmovsxbd	%xmm6, %ymm15
1000063b6: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000063bb: c4 42 65 40 ff              	vpmulld	%ymm15, %ymm3, %ymm15
1000063c0: c4 c1 45 fe ff              	vpaddd	%ymm15, %ymm7, %ymm7
1000063c5: c4 41 1d fe c9              	vpaddd	%ymm9, %ymm12, %ymm9
1000063ca: c4 e3 fd 00 c9 db           	vpermq	$219, %ymm1, %ymm1
1000063d0: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000063d5: c4 e2 65 40 c9              	vpmulld	%ymm1, %ymm3, %ymm1
1000063da: c5 dd fe c9                 	vpaddd	%ymm1, %ymm4, %ymm1
1000063de: c4 c1 0d fe e5              	vpaddd	%ymm13, %ymm14, %ymm4
1000063e3: c4 e3 7d 39 f6 01           	vextracti128	$1, %ymm6, %xmm6
1000063e9: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
1000063ee: c4 e2 65 40 f6              	vpmulld	%ymm6, %ymm3, %ymm6
1000063f3: c5 d5 fe ee                 	vpaddd	%ymm6, %ymm5, %ymm5
1000063f7: c4 e2 79 00 05 c0 0d 00 00  	vpshufb	3520(%rip), %xmm0, %xmm0
100006400: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006405: c4 e2 65 40 c0              	vpmulld	%ymm0, %ymm3, %ymm0
10000640a: c5 ed fe c0                 	vpaddd	%ymm0, %ymm2, %ymm0
10000640e: 48 8b 44 24 18              	movq	24(%rsp), %rax
100006413: c4 e2 79 78 10              	vpbroadcastb	(%rax), %xmm2
100006418: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000641d: c5 c5 fe da                 	vpaddd	%ymm2, %ymm7, %ymm3
100006421: c5 bd fe db                 	vpaddd	%ymm3, %ymm8, %ymm3
100006425: c5 f5 fe ca                 	vpaddd	%ymm2, %ymm1, %ymm1
100006429: c5 a5 fe c9                 	vpaddd	%ymm1, %ymm11, %ymm1
10000642d: c5 b5 fe e4                 	vpaddd	%ymm4, %ymm9, %ymm4
100006431: c5 d5 fe ea                 	vpaddd	%ymm2, %ymm5, %ymm5
100006435: c5 fd fe c2                 	vpaddd	%ymm2, %ymm0, %ymm0
100006439: c5 ad fe c0                 	vpaddd	%ymm0, %ymm10, %ymm0
10000643d: c5 fd 6f b4 24 60 02 00 00  	vmovdqa	608(%rsp), %ymm6
100006446: c4 e2 75 40 ce              	vpmulld	%ymm6, %ymm1, %ymm1
10000644b: c5 dd fe d5                 	vpaddd	%ymm5, %ymm4, %ymm2
10000644f: c4 e2 65 40 de              	vpmulld	%ymm6, %ymm3, %ymm3
100006454: c4 e2 7d 40 c6              	vpmulld	%ymm6, %ymm0, %ymm0
100006459: c4 e2 6d 40 d6              	vpmulld	%ymm6, %ymm2, %ymm2
10000645e: c5 dd 72 e3 1f              	vpsrad	$31, %ymm3, %ymm4
100006463: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
100006468: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
10000646c: c5 dd 72 e1 1f              	vpsrad	$31, %ymm1, %ymm4
100006471: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
100006476: c5 e5 72 e3 0e              	vpsrad	$14, %ymm3, %ymm3
10000647b: c5 f5 fe cc                 	vpaddd	%ymm4, %ymm1, %ymm1
10000647f: c5 f5 72 e1 0e              	vpsrad	$14, %ymm1, %ymm1
100006484: c5 dd 72 e2 1f              	vpsrad	$31, %ymm2, %ymm4
100006489: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
10000648e: c5 ed fe d4                 	vpaddd	%ymm4, %ymm2, %ymm2
100006492: c5 ed 72 e2 0e              	vpsrad	$14, %ymm2, %ymm2
100006497: c5 dd 72 e0 1f              	vpsrad	$31, %ymm0, %ymm4
10000649c: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
1000064a1: c5 fd fe c4                 	vpaddd	%ymm4, %ymm0, %ymm0
1000064a5: c5 fd 72 e0 0e              	vpsrad	$14, %ymm0, %ymm0
1000064aa: c4 e2 7d 58 25 2d 28 00 00  	vpbroadcastd	10285(%rip), %ymm4
1000064b3: c4 e2 6d 39 d4              	vpminsd	%ymm4, %ymm2, %ymm2
1000064b8: c4 e2 75 39 cc              	vpminsd	%ymm4, %ymm1, %ymm1
1000064bd: c4 e2 65 39 dc              	vpminsd	%ymm4, %ymm3, %ymm3
1000064c2: c4 e2 7d 39 e4              	vpminsd	%ymm4, %ymm0, %ymm4
1000064c7: c4 e2 7d 58 2d 14 28 00 00  	vpbroadcastd	10260(%rip), %ymm5
1000064d0: c4 e2 75 3d c5              	vpmaxsd	%ymm5, %ymm1, %ymm0
1000064d5: c4 e2 6d 3d cd              	vpmaxsd	%ymm5, %ymm2, %ymm1
1000064da: c5 f5 6b c0                 	vpackssdw	%ymm0, %ymm1, %ymm0
1000064de: c4 e2 65 3d cd              	vpmaxsd	%ymm5, %ymm3, %ymm1
1000064e3: c4 e2 5d 3d d5              	vpmaxsd	%ymm5, %ymm4, %ymm2
1000064e8: c5 f5 6b ca                 	vpackssdw	%ymm2, %ymm1, %ymm1
1000064ec: c5 fd 6f b4 24 40 02 00 00  	vmovdqa	576(%rsp), %ymm6
1000064f5: c5 ed 73 d6 01              	vpsrlq	$1, %ymm6, %ymm2
1000064fa: c5 fd 6f ac 24 a0 02 00 00  	vmovdqa	672(%rsp), %ymm5
100006503: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006507: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000650c: c5 fd 6f a4 24 80 02 00 00  	vmovdqa	640(%rsp), %ymm4
100006515: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006519: c4 c1 f9 7e d2              	vmovq	%xmm2, %r10
10000651e: c4 e3 f9 16 d0 01           	vpextrq	$1, %xmm2, %rax
100006524: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000652a: c4 c1 f9 7e d0              	vmovq	%xmm2, %r8
10000652f: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
100006535: c5 fd 6f bc 24 20 02 00 00  	vmovdqa	544(%rsp), %ymm7
10000653e: c5 ed 73 d7 01              	vpsrlq	$1, %ymm7, %ymm2
100006543: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006547: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000654c: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006550: c4 c1 f9 7e d5              	vmovq	%xmm2, %r13
100006555: c4 c3 f9 16 d6 01           	vpextrq	$1, %xmm2, %r14
10000655b: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006561: c4 e1 f9 7e d6              	vmovq	%xmm2, %rsi
100006566: c4 e3 f9 16 d7 01           	vpextrq	$1, %xmm2, %rdi
10000656c: c5 7d 6f 84 24 00 02 00 00  	vmovdqa	512(%rsp), %ymm8
100006575: c4 c1 6d 73 d0 01           	vpsrlq	$1, %ymm8, %ymm2
10000657b: c4 e3 fd 00 c0 d8           	vpermq	$216, %ymm0, %ymm0
100006581: c4 e3 fd 00 c9 d8           	vpermq	$216, %ymm1, %ymm1
100006587: c5 f5 63 c0                 	vpacksswb	%ymm0, %ymm1, %ymm0
10000658b: c5 7d 6f 8c 24 e0 01 00 00  	vmovdqa	480(%rsp), %ymm9
100006594: c4 c1 65 73 d1 01           	vpsrlq	$1, %ymm9, %ymm3
10000659a: c5 ed d4 cd                 	vpaddq	%ymm5, %ymm2, %ymm1
10000659e: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
1000065a3: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000065a7: c4 e3 f9 16 8c 24 40 01 00 00 01    	vpextrq	$1, %xmm1, 320(%rsp)
1000065b2: c4 e1 f9 7e ca              	vmovq	%xmm1, %rdx
1000065b7: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
1000065bd: c5 f9 d6 8c 24 20 01 00 00  	vmovq	%xmm1, 288(%rsp)
1000065c6: c4 c3 f9 16 cc 01           	vpextrq	$1, %xmm1, %r12
1000065cc: c5 7d 6f 94 24 c0 01 00 00  	vmovdqa	448(%rsp), %ymm10
1000065d5: c4 c1 75 73 d2 01           	vpsrlq	$1, %ymm10, %ymm1
1000065db: c5 e5 d4 d5                 	vpaddq	%ymm5, %ymm3, %ymm2
1000065df: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000065e4: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000065e8: c4 83 79 14 04 17 00        	vpextrb	$0, %xmm0, (%r15,%r10)
1000065ef: c4 e1 f9 7e d1              	vmovq	%xmm2, %rcx
1000065f4: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
1000065fa: c4 c3 79 14 04 07 01        	vpextrb	$1, %xmm0, (%r15,%rax)
100006601: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006607: c4 e3 f9 16 94 24 00 01 00 00 01    	vpextrq	$1, %xmm2, 256(%rsp)
100006612: c4 83 79 14 04 07 02        	vpextrb	$2, %xmm0, (%r15,%r8)
100006619: c4 e1 f9 7e d3              	vmovq	%xmm2, %rbx
10000661e: c5 fd 6f 9c 24 a0 01 00 00  	vmovdqa	416(%rsp), %ymm3
100006627: c5 ed 73 d3 01              	vpsrlq	$1, %ymm3, %ymm2
10000662c: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006630: c5 f5 d4 cd                 	vpaddq	%ymm5, %ymm1, %ymm1
100006634: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
100006639: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000663e: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006642: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100006646: c4 83 79 14 04 0f 03        	vpextrb	$3, %xmm0, (%r15,%r9)
10000664d: c4 c1 f9 7e c8              	vmovq	%xmm1, %r8
100006652: c4 83 79 14 04 2f 04        	vpextrb	$4, %xmm0, (%r15,%r13)
100006659: c4 e3 f9 16 8c 24 c0 00 00 00 01    	vpextrq	$1, %xmm1, 192(%rsp)
100006664: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
10000666a: c4 83 79 14 04 37 05        	vpextrb	$5, %xmm0, (%r15,%r14)
100006671: c4 c3 79 14 04 37 06        	vpextrb	$6, %xmm0, (%r15,%rsi)
100006678: c4 c1 f9 7e ce              	vmovq	%xmm1, %r14
10000667d: c4 e3 f9 16 8c 24 e0 00 00 00 01    	vpextrq	$1, %xmm1, 224(%rsp)
100006688: c4 c3 79 14 04 3f 07        	vpextrb	$7, %xmm0, (%r15,%rdi)
10000668f: c4 e3 f9 16 94 24 a0 00 00 00 01    	vpextrq	$1, %xmm2, 160(%rsp)
10000669a: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
1000066a0: c4 c3 79 14 0c 17 00        	vpextrb	$0, %xmm1, (%r15,%rdx)
1000066a7: c4 e1 f9 7e d7              	vmovq	%xmm2, %rdi
1000066ac: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000066b2: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
1000066ba: c4 c3 79 14 0c 07 01        	vpextrb	$1, %xmm1, (%r15,%rax)
1000066c1: c4 c1 f9 7e d5              	vmovq	%xmm2, %r13
1000066c6: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
1000066ce: c4 c3 79 14 0c 07 02        	vpextrb	$2, %xmm1, (%r15,%rax)
1000066d5: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
1000066db: c5 7d 6f 9c 24 80 01 00 00  	vmovdqa	384(%rsp), %ymm11
1000066e4: c4 c1 6d 73 d3 01           	vpsrlq	$1, %ymm11, %ymm2
1000066ea: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
1000066ee: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000066f3: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000066f7: c4 83 79 14 0c 27 03        	vpextrb	$3, %xmm1, (%r15,%r12)
1000066fe: c4 e1 f9 7e d2              	vmovq	%xmm2, %rdx
100006703: c4 c3 79 14 0c 0f 04        	vpextrb	$4, %xmm1, (%r15,%rcx)
10000670a: c4 e3 f9 16 d1 01           	vpextrq	$1, %xmm2, %rcx
100006710: c4 83 79 14 0c 17 05        	vpextrb	$5, %xmm1, (%r15,%r10)
100006717: c4 c3 79 14 0c 1f 06        	vpextrb	$6, %xmm1, (%r15,%rbx)
10000671e: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006724: c4 e1 f9 7e d3              	vmovq	%xmm2, %rbx
100006729: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
10000672f: c5 7d 6f a4 24 60 01 00 00  	vmovdqa	352(%rsp), %ymm12
100006738: c4 c1 6d 73 d4 01           	vpsrlq	$1, %ymm12, %ymm2
10000673e: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006742: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006747: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
10000674b: 48 8b 84 24 00 01 00 00     	movq	256(%rsp), %rax
100006753: c4 c3 79 14 0c 07 07        	vpextrb	$7, %xmm1, (%r15,%rax)
10000675a: c4 e1 f9 7e d0              	vmovq	%xmm2, %rax
10000675f: c4 83 79 14 04 07 08        	vpextrb	$8, %xmm0, (%r15,%r8)
100006766: c4 c3 f9 16 d0 01           	vpextrq	$1, %xmm2, %r8
10000676c: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006772: 48 8b b4 24 c0 00 00 00     	movq	192(%rsp), %rsi
10000677a: c4 c3 79 14 04 37 09        	vpextrb	$9, %xmm0, (%r15,%rsi)
100006781: c4 83 79 14 04 37 0a        	vpextrb	$10, %xmm0, (%r15,%r14)
100006788: c4 c1 f9 7e d6              	vmovq	%xmm2, %r14
10000678d: c4 c3 f9 16 d4 01           	vpextrq	$1, %xmm2, %r12
100006793: c5 fd 6f 15 45 0b 00 00     	vmovdqa	2885(%rip), %ymm2
10000679b: 48 8b b4 24 e0 00 00 00     	movq	224(%rsp), %rsi
1000067a3: c4 c3 79 14 04 37 0b        	vpextrb	$11, %xmm0, (%r15,%rsi)
1000067aa: c4 c3 79 14 04 3f 0c        	vpextrb	$12, %xmm0, (%r15,%rdi)
1000067b1: 48 8b b4 24 a0 00 00 00     	movq	160(%rsp), %rsi
1000067b9: c4 c3 79 14 04 37 0d        	vpextrb	$13, %xmm0, (%r15,%rsi)
1000067c0: c4 83 79 14 04 2f 0e        	vpextrb	$14, %xmm0, (%r15,%r13)
1000067c7: c4 83 79 14 04 0f 0f        	vpextrb	$15, %xmm0, (%r15,%r9)
1000067ce: c4 c3 79 14 0c 17 08        	vpextrb	$8, %xmm1, (%r15,%rdx)
1000067d5: c4 c3 79 14 0c 0f 09        	vpextrb	$9, %xmm1, (%r15,%rcx)
1000067dc: c4 c3 79 14 0c 1f 0a        	vpextrb	$10, %xmm1, (%r15,%rbx)
1000067e3: c4 83 79 14 0c 17 0b        	vpextrb	$11, %xmm1, (%r15,%r10)
1000067ea: c4 c3 79 14 0c 07 0c        	vpextrb	$12, %xmm1, (%r15,%rax)
1000067f1: c4 83 79 14 0c 07 0d        	vpextrb	$13, %xmm1, (%r15,%r8)
1000067f8: c4 83 79 14 0c 37 0e        	vpextrb	$14, %xmm1, (%r15,%r14)
1000067ff: c4 83 79 14 0c 27 0f        	vpextrb	$15, %xmm1, (%r15,%r12)
100006806: c4 e2 7d 59 05 d9 24 00 00  	vpbroadcastq	9433(%rip), %ymm0
10000680f: c5 cd d4 f0                 	vpaddq	%ymm0, %ymm6, %ymm6
100006813: c5 fd 7f b4 24 40 02 00 00  	vmovdqa	%ymm6, 576(%rsp)
10000681c: c5 c5 d4 f8                 	vpaddq	%ymm0, %ymm7, %ymm7
100006820: c5 fd 7f bc 24 20 02 00 00  	vmovdqa	%ymm7, 544(%rsp)
100006829: c5 3d d4 c0                 	vpaddq	%ymm0, %ymm8, %ymm8
10000682d: c5 7d 7f 84 24 00 02 00 00  	vmovdqa	%ymm8, 512(%rsp)
100006836: c5 35 d4 c8                 	vpaddq	%ymm0, %ymm9, %ymm9
10000683a: c5 7d 7f 8c 24 e0 01 00 00  	vmovdqa	%ymm9, 480(%rsp)
100006843: c5 2d d4 d0                 	vpaddq	%ymm0, %ymm10, %ymm10
100006847: c5 7d 7f 94 24 c0 01 00 00  	vmovdqa	%ymm10, 448(%rsp)
100006850: c5 e5 d4 d8                 	vpaddq	%ymm0, %ymm3, %ymm3
100006854: c5 fd 7f 9c 24 a0 01 00 00  	vmovdqa	%ymm3, 416(%rsp)
10000685d: c5 25 d4 d8                 	vpaddq	%ymm0, %ymm11, %ymm11
100006861: c5 7d 7f 9c 24 80 01 00 00  	vmovdqa	%ymm11, 384(%rsp)
10000686a: c5 1d d4 e0                 	vpaddq	%ymm0, %ymm12, %ymm12
10000686e: c5 7d 7f a4 24 60 01 00 00  	vmovdqa	%ymm12, 352(%rsp)
100006877: 49 83 c3 20                 	addq	$32, %r11
10000687b: 49 81 fb e0 00 00 00        	cmpq	$224, %r11
100006882: 0f 85 68 f6 ff ff           	jne	-2456 <__ZN11LineNetwork7forwardEv+0x840>
100006888: ba c0 01 00 00              	movl	$448, %edx
10000688d: 44 8b 44 24 14              	movl	20(%rsp), %r8d
100006892: 48 8b 74 24 58              	movq	88(%rsp), %rsi
100006897: 4c 8b 6c 24 68              	movq	104(%rsp), %r13
10000689c: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
1000068a1: eb 0f                       	jmp	15 <__ZN11LineNetwork7forwardEv+0x1202>
1000068a3: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000068ad: 0f 1f 00                    	nopl	(%rax)
1000068b0: 31 d2                       	xorl	%edx, %edx
1000068b2: 48 83 44 24 08 02           	addq	$2, 8(%rsp)
1000068b8: 48 89 d0                    	movq	%rdx, %rax
1000068bb: 48 d1 e8                    	shrq	%rax
1000068be: 4c 8b b4 24 98 00 00 00     	movq	152(%rsp), %r14
1000068c6: 4c 01 f0                    	addq	%r14, %rax
1000068c9: 4c 8d 0c c7                 	leaq	(%rdi,%rax,8), %r9
1000068cd: 4c 8b 54 24 20              	movq	32(%rsp), %r10
1000068d2: 4c 8b 5c 24 18              	movq	24(%rsp), %r11
1000068d7: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x1248>
1000068d9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
1000068e0: 41 88 09                    	movb	%cl, (%r9)
1000068e3: 48 83 c2 02                 	addq	$2, %rdx
1000068e7: 49 83 c1 08                 	addq	$8, %r9
1000068eb: 48 81 fa fd 01 00 00        	cmpq	$509, %rdx
1000068f2: 0f 83 78 f4 ff ff           	jae	-2952 <__ZN11LineNetwork7forwardEv+0x6c0>
1000068f8: 4c 8b 64 24 30              	movq	48(%rsp), %r12
1000068fd: 41 0f be 8c 14 fe fb ff ff  	movsbl	-1026(%r12,%rdx), %ecx
100006906: 41 0f be 02                 	movsbl	(%r10), %eax
10000690a: 0f af c1                    	imull	%ecx, %eax
10000690d: 41 0f be 8c 14 ff fb ff ff  	movsbl	-1025(%r12,%rdx), %ecx
100006916: 41 0f be 5a 01              	movsbl	1(%r10), %ebx
10000691b: 0f af d9                    	imull	%ecx, %ebx
10000691e: 01 c3                       	addl	%eax, %ebx
100006920: 41 0f be 8c 14 00 fc ff ff  	movsbl	-1024(%r12,%rdx), %ecx
100006929: 41 0f be 42 02              	movsbl	2(%r10), %eax
10000692e: 0f af c1                    	imull	%ecx, %eax
100006931: 01 d8                       	addl	%ebx, %eax
100006933: 41 0f be 8c 14 fe fd ff ff  	movsbl	-514(%r12,%rdx), %ecx
10000693c: 41 0f be 5a 03              	movsbl	3(%r10), %ebx
100006941: 0f af d9                    	imull	%ecx, %ebx
100006944: 01 c3                       	addl	%eax, %ebx
100006946: 41 0f be 8c 14 ff fd ff ff  	movsbl	-513(%r12,%rdx), %ecx
10000694f: 41 0f be 42 04              	movsbl	4(%r10), %eax
100006954: 0f af c1                    	imull	%ecx, %eax
100006957: 01 d8                       	addl	%ebx, %eax
100006959: 41 0f be 8c 14 00 fe ff ff  	movsbl	-512(%r12,%rdx), %ecx
100006962: 41 0f be 5a 05              	movsbl	5(%r10), %ebx
100006967: 0f af d9                    	imull	%ecx, %ebx
10000696a: 01 c3                       	addl	%eax, %ebx
10000696c: 41 0f be 4c 14 fe           	movsbl	-2(%r12,%rdx), %ecx
100006972: 41 0f be 42 06              	movsbl	6(%r10), %eax
100006977: 0f af c1                    	imull	%ecx, %eax
10000697a: 01 d8                       	addl	%ebx, %eax
10000697c: 41 0f be 4c 14 ff           	movsbl	-1(%r12,%rdx), %ecx
100006982: 41 0f be 5a 07              	movsbl	7(%r10), %ebx
100006987: 0f af d9                    	imull	%ecx, %ebx
10000698a: 01 c3                       	addl	%eax, %ebx
10000698c: 41 0f be 0c 14              	movsbl	(%r12,%rdx), %ecx
100006991: 41 0f be 42 08              	movsbl	8(%r10), %eax
100006996: 0f af c1                    	imull	%ecx, %eax
100006999: 01 d8                       	addl	%ebx, %eax
10000699b: 41 0f be 1b                 	movsbl	(%r11), %ebx
10000699f: 01 c3                       	addl	%eax, %ebx
1000069a1: 41 0f af d8                 	imull	%r8d, %ebx
1000069a5: 89 d9                       	movl	%ebx, %ecx
1000069a7: c1 f9 1f                    	sarl	$31, %ecx
1000069aa: c1 e9 12                    	shrl	$18, %ecx
1000069ad: 01 d9                       	addl	%ebx, %ecx
1000069af: c1 f9 0e                    	sarl	$14, %ecx
1000069b2: 81 f9 80 00 00 00           	cmpl	$128, %ecx
1000069b8: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x130f>
1000069ba: b9 7f 00 00 00              	movl	$127, %ecx
1000069bf: 83 f9 81                    	cmpl	$-127, %ecx
1000069c2: 0f 8f 18 ff ff ff           	jg	-232 <__ZN11LineNetwork7forwardEv+0x1230>
1000069c8: b9 81 00 00 00              	movl	$129, %ecx
1000069cd: e9 0e ff ff ff              	jmp	-242 <__ZN11LineNetwork7forwardEv+0x1230>
1000069d2: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
1000069d6: 5b                          	popq	%rbx
1000069d7: 41 5c                       	popq	%r12
1000069d9: 41 5d                       	popq	%r13
1000069db: 41 5e                       	popq	%r14
1000069dd: 41 5f                       	popq	%r15
1000069df: 5d                          	popq	%rbp
1000069e0: c5 f8 77                    	vzeroupper
1000069e3: c3                          	retq
1000069e4: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000069ee: 66 90                       	nop
1000069f0: 55                          	pushq	%rbp
1000069f1: 48 89 e5                    	movq	%rsp, %rbp
1000069f4: 5d                          	popq	%rbp
1000069f5: e9 16 df ff ff              	jmp	-8426 <__ZN14ModelInterfaceD2Ev>
1000069fa: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100006a00: 55                          	pushq	%rbp
100006a01: 48 89 e5                    	movq	%rsp, %rbp
100006a04: 53                          	pushq	%rbx
100006a05: 50                          	pushq	%rax
100006a06: 48 89 fb                    	movq	%rdi, %rbx
100006a09: e8 02 df ff ff              	callq	-8446 <__ZN14ModelInterfaceD2Ev>
100006a0e: 48 89 df                    	movq	%rbx, %rdi
100006a11: 48 83 c4 08                 	addq	$8, %rsp
100006a15: 5b                          	popq	%rbx
100006a16: 5d                          	popq	%rbp
100006a17: e9 76 04 00 00              	jmp	1142 <dyld_stub_binder+0x100006e92>
100006a1c: 0f 1f 40 00                 	nopl	(%rax)
100006a20: 55                          	pushq	%rbp
100006a21: 48 89 e5                    	movq	%rsp, %rbp
100006a24: c4 e2 7d 21 06              	vpmovsxbd	(%rsi), %ymm0
100006a29: c4 e2 7d 21 4e 08           	vpmovsxbd	8(%rsi), %ymm1
100006a2f: c4 e2 7d 21 17              	vpmovsxbd	(%rdi), %ymm2
100006a34: c4 e2 7d 40 c2              	vpmulld	%ymm2, %ymm0, %ymm0
100006a39: c4 e2 7d 21 57 08           	vpmovsxbd	8(%rdi), %ymm2
100006a3f: c4 e2 75 40 ca              	vpmulld	%ymm2, %ymm1, %ymm1
100006a44: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006a48: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006a4e: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006a52: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006a57: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006a5b: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006a60: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006a64: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006a68: 0f be 4f 10                 	movsbl	16(%rdi), %ecx
100006a6c: 0f be 56 10                 	movsbl	16(%rsi), %edx
100006a70: 0f af d1                    	imull	%ecx, %edx
100006a73: 01 c2                       	addl	%eax, %edx
100006a75: 0f be 47 11                 	movsbl	17(%rdi), %eax
100006a79: 0f be 4e 11                 	movsbl	17(%rsi), %ecx
100006a7d: 0f af c8                    	imull	%eax, %ecx
100006a80: 01 d1                       	addl	%edx, %ecx
100006a82: 0f be 47 12                 	movsbl	18(%rdi), %eax
100006a86: 0f be 56 12                 	movsbl	18(%rsi), %edx
100006a8a: 0f af d0                    	imull	%eax, %edx
100006a8d: 01 ca                       	addl	%ecx, %edx
100006a8f: 0f be 47 13                 	movsbl	19(%rdi), %eax
100006a93: 0f be 4e 13                 	movsbl	19(%rsi), %ecx
100006a97: 0f af c8                    	imull	%eax, %ecx
100006a9a: 01 d1                       	addl	%edx, %ecx
100006a9c: 0f be 47 14                 	movsbl	20(%rdi), %eax
100006aa0: 0f be 56 14                 	movsbl	20(%rsi), %edx
100006aa4: 0f af d0                    	imull	%eax, %edx
100006aa7: 01 ca                       	addl	%ecx, %edx
100006aa9: 0f be 47 15                 	movsbl	21(%rdi), %eax
100006aad: 0f be 4e 15                 	movsbl	21(%rsi), %ecx
100006ab1: 0f af c8                    	imull	%eax, %ecx
100006ab4: 01 d1                       	addl	%edx, %ecx
100006ab6: 0f be 47 16                 	movsbl	22(%rdi), %eax
100006aba: 0f be 56 16                 	movsbl	22(%rsi), %edx
100006abe: 0f af d0                    	imull	%eax, %edx
100006ac1: 01 ca                       	addl	%ecx, %edx
100006ac3: 0f be 4f 17                 	movsbl	23(%rdi), %ecx
100006ac7: 0f be 46 17                 	movsbl	23(%rsi), %eax
100006acb: 0f af c1                    	imull	%ecx, %eax
100006ace: 01 d0                       	addl	%edx, %eax
100006ad0: 5d                          	popq	%rbp
100006ad1: c5 f8 77                    	vzeroupper
100006ad4: c3                          	retq
100006ad5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100006adf: 90                          	nop
100006ae0: 55                          	pushq	%rbp
100006ae1: 48 89 e5                    	movq	%rsp, %rbp
100006ae4: 0f be 06                    	movsbl	(%rsi), %eax
100006ae7: 0f be 0f                    	movsbl	(%rdi), %ecx
100006aea: 0f af c8                    	imull	%eax, %ecx
100006aed: 0f be 46 01                 	movsbl	1(%rsi), %eax
100006af1: 0f be 57 01                 	movsbl	1(%rdi), %edx
100006af5: 0f af d0                    	imull	%eax, %edx
100006af8: 01 ca                       	addl	%ecx, %edx
100006afa: 0f be 46 02                 	movsbl	2(%rsi), %eax
100006afe: 0f be 4f 02                 	movsbl	2(%rdi), %ecx
100006b02: 0f af c8                    	imull	%eax, %ecx
100006b05: 01 d1                       	addl	%edx, %ecx
100006b07: 0f be 46 03                 	movsbl	3(%rsi), %eax
100006b0b: 0f be 57 03                 	movsbl	3(%rdi), %edx
100006b0f: 0f af d0                    	imull	%eax, %edx
100006b12: 01 ca                       	addl	%ecx, %edx
100006b14: 0f be 46 04                 	movsbl	4(%rsi), %eax
100006b18: 0f be 4f 04                 	movsbl	4(%rdi), %ecx
100006b1c: 0f af c8                    	imull	%eax, %ecx
100006b1f: 01 d1                       	addl	%edx, %ecx
100006b21: 0f be 46 05                 	movsbl	5(%rsi), %eax
100006b25: 0f be 57 05                 	movsbl	5(%rdi), %edx
100006b29: 0f af d0                    	imull	%eax, %edx
100006b2c: 01 ca                       	addl	%ecx, %edx
100006b2e: 0f be 46 06                 	movsbl	6(%rsi), %eax
100006b32: 0f be 4f 06                 	movsbl	6(%rdi), %ecx
100006b36: 0f af c8                    	imull	%eax, %ecx
100006b39: 01 d1                       	addl	%edx, %ecx
100006b3b: 0f be 46 07                 	movsbl	7(%rsi), %eax
100006b3f: 0f be 57 07                 	movsbl	7(%rdi), %edx
100006b43: 0f af d0                    	imull	%eax, %edx
100006b46: 01 ca                       	addl	%ecx, %edx
100006b48: 0f be 46 08                 	movsbl	8(%rsi), %eax
100006b4c: 0f be 4f 08                 	movsbl	8(%rdi), %ecx
100006b50: 0f af c8                    	imull	%eax, %ecx
100006b53: 01 d1                       	addl	%edx, %ecx
100006b55: 0f be 46 09                 	movsbl	9(%rsi), %eax
100006b59: 0f be 57 09                 	movsbl	9(%rdi), %edx
100006b5d: 0f af d0                    	imull	%eax, %edx
100006b60: 01 ca                       	addl	%ecx, %edx
100006b62: 0f be 46 0a                 	movsbl	10(%rsi), %eax
100006b66: 0f be 4f 0a                 	movsbl	10(%rdi), %ecx
100006b6a: 0f af c8                    	imull	%eax, %ecx
100006b6d: 01 d1                       	addl	%edx, %ecx
100006b6f: 0f be 46 0b                 	movsbl	11(%rsi), %eax
100006b73: 0f be 57 0b                 	movsbl	11(%rdi), %edx
100006b77: 0f af d0                    	imull	%eax, %edx
100006b7a: 01 ca                       	addl	%ecx, %edx
100006b7c: 0f be 46 0c                 	movsbl	12(%rsi), %eax
100006b80: 0f be 4f 0c                 	movsbl	12(%rdi), %ecx
100006b84: 0f af c8                    	imull	%eax, %ecx
100006b87: 01 d1                       	addl	%edx, %ecx
100006b89: 0f be 46 0d                 	movsbl	13(%rsi), %eax
100006b8d: 0f be 57 0d                 	movsbl	13(%rdi), %edx
100006b91: 0f af d0                    	imull	%eax, %edx
100006b94: 01 ca                       	addl	%ecx, %edx
100006b96: 0f be 46 0e                 	movsbl	14(%rsi), %eax
100006b9a: 0f be 4f 0e                 	movsbl	14(%rdi), %ecx
100006b9e: 0f af c8                    	imull	%eax, %ecx
100006ba1: 01 d1                       	addl	%edx, %ecx
100006ba3: 0f be 46 0f                 	movsbl	15(%rsi), %eax
100006ba7: 0f be 57 0f                 	movsbl	15(%rdi), %edx
100006bab: 0f af d0                    	imull	%eax, %edx
100006bae: 01 ca                       	addl	%ecx, %edx
100006bb0: 0f be 46 10                 	movsbl	16(%rsi), %eax
100006bb4: 0f be 4f 10                 	movsbl	16(%rdi), %ecx
100006bb8: 0f af c8                    	imull	%eax, %ecx
100006bbb: 01 d1                       	addl	%edx, %ecx
100006bbd: 0f be 46 11                 	movsbl	17(%rsi), %eax
100006bc1: 0f be 57 11                 	movsbl	17(%rdi), %edx
100006bc5: 0f af d0                    	imull	%eax, %edx
100006bc8: 01 ca                       	addl	%ecx, %edx
100006bca: 0f be 46 12                 	movsbl	18(%rsi), %eax
100006bce: 0f be 4f 12                 	movsbl	18(%rdi), %ecx
100006bd2: 0f af c8                    	imull	%eax, %ecx
100006bd5: 01 d1                       	addl	%edx, %ecx
100006bd7: 0f be 46 13                 	movsbl	19(%rsi), %eax
100006bdb: 0f be 57 13                 	movsbl	19(%rdi), %edx
100006bdf: 0f af d0                    	imull	%eax, %edx
100006be2: 01 ca                       	addl	%ecx, %edx
100006be4: 0f be 46 14                 	movsbl	20(%rsi), %eax
100006be8: 0f be 4f 14                 	movsbl	20(%rdi), %ecx
100006bec: 0f af c8                    	imull	%eax, %ecx
100006bef: 01 d1                       	addl	%edx, %ecx
100006bf1: 0f be 46 15                 	movsbl	21(%rsi), %eax
100006bf5: 0f be 57 15                 	movsbl	21(%rdi), %edx
100006bf9: 0f af d0                    	imull	%eax, %edx
100006bfc: 01 ca                       	addl	%ecx, %edx
100006bfe: 0f be 46 16                 	movsbl	22(%rsi), %eax
100006c02: 0f be 4f 16                 	movsbl	22(%rdi), %ecx
100006c06: 0f af c8                    	imull	%eax, %ecx
100006c09: 01 d1                       	addl	%edx, %ecx
100006c0b: 0f be 46 17                 	movsbl	23(%rsi), %eax
100006c0f: 0f be 57 17                 	movsbl	23(%rdi), %edx
100006c13: 0f af d0                    	imull	%eax, %edx
100006c16: 01 ca                       	addl	%ecx, %edx
100006c18: 0f be 46 18                 	movsbl	24(%rsi), %eax
100006c1c: 0f be 4f 18                 	movsbl	24(%rdi), %ecx
100006c20: 0f af c8                    	imull	%eax, %ecx
100006c23: 01 d1                       	addl	%edx, %ecx
100006c25: 0f be 46 19                 	movsbl	25(%rsi), %eax
100006c29: 0f be 57 19                 	movsbl	25(%rdi), %edx
100006c2d: 0f af d0                    	imull	%eax, %edx
100006c30: 01 ca                       	addl	%ecx, %edx
100006c32: 0f be 46 1a                 	movsbl	26(%rsi), %eax
100006c36: 0f be 4f 1a                 	movsbl	26(%rdi), %ecx
100006c3a: 0f af c8                    	imull	%eax, %ecx
100006c3d: 01 d1                       	addl	%edx, %ecx
100006c3f: 0f be 46 1b                 	movsbl	27(%rsi), %eax
100006c43: 0f be 57 1b                 	movsbl	27(%rdi), %edx
100006c47: 0f af d0                    	imull	%eax, %edx
100006c4a: 01 ca                       	addl	%ecx, %edx
100006c4c: 0f be 46 1c                 	movsbl	28(%rsi), %eax
100006c50: 0f be 4f 1c                 	movsbl	28(%rdi), %ecx
100006c54: 0f af c8                    	imull	%eax, %ecx
100006c57: 01 d1                       	addl	%edx, %ecx
100006c59: 0f be 46 1d                 	movsbl	29(%rsi), %eax
100006c5d: 0f be 57 1d                 	movsbl	29(%rdi), %edx
100006c61: 0f af d0                    	imull	%eax, %edx
100006c64: 01 ca                       	addl	%ecx, %edx
100006c66: 0f be 46 1e                 	movsbl	30(%rsi), %eax
100006c6a: 0f be 4f 1e                 	movsbl	30(%rdi), %ecx
100006c6e: 0f af c8                    	imull	%eax, %ecx
100006c71: 01 d1                       	addl	%edx, %ecx
100006c73: 0f be 46 1f                 	movsbl	31(%rsi), %eax
100006c77: 0f be 57 1f                 	movsbl	31(%rdi), %edx
100006c7b: 0f af d0                    	imull	%eax, %edx
100006c7e: 01 ca                       	addl	%ecx, %edx
100006c80: 0f be 47 20                 	movsbl	32(%rdi), %eax
100006c84: 0f be 4e 20                 	movsbl	32(%rsi), %ecx
100006c88: 0f af c8                    	imull	%eax, %ecx
100006c8b: 01 d1                       	addl	%edx, %ecx
100006c8d: 0f be 47 21                 	movsbl	33(%rdi), %eax
100006c91: 0f be 56 21                 	movsbl	33(%rsi), %edx
100006c95: 0f af d0                    	imull	%eax, %edx
100006c98: 01 ca                       	addl	%ecx, %edx
100006c9a: 0f be 47 22                 	movsbl	34(%rdi), %eax
100006c9e: 0f be 4e 22                 	movsbl	34(%rsi), %ecx
100006ca2: 0f af c8                    	imull	%eax, %ecx
100006ca5: 01 d1                       	addl	%edx, %ecx
100006ca7: 0f be 47 23                 	movsbl	35(%rdi), %eax
100006cab: 0f be 56 23                 	movsbl	35(%rsi), %edx
100006caf: 0f af d0                    	imull	%eax, %edx
100006cb2: 01 ca                       	addl	%ecx, %edx
100006cb4: 0f be 47 24                 	movsbl	36(%rdi), %eax
100006cb8: 0f be 4e 24                 	movsbl	36(%rsi), %ecx
100006cbc: 0f af c8                    	imull	%eax, %ecx
100006cbf: 01 d1                       	addl	%edx, %ecx
100006cc1: 0f be 47 25                 	movsbl	37(%rdi), %eax
100006cc5: 0f be 56 25                 	movsbl	37(%rsi), %edx
100006cc9: 0f af d0                    	imull	%eax, %edx
100006ccc: 01 ca                       	addl	%ecx, %edx
100006cce: 0f be 47 26                 	movsbl	38(%rdi), %eax
100006cd2: 0f be 4e 26                 	movsbl	38(%rsi), %ecx
100006cd6: 0f af c8                    	imull	%eax, %ecx
100006cd9: 01 d1                       	addl	%edx, %ecx
100006cdb: 0f be 47 27                 	movsbl	39(%rdi), %eax
100006cdf: 0f be 56 27                 	movsbl	39(%rsi), %edx
100006ce3: 0f af d0                    	imull	%eax, %edx
100006ce6: 01 ca                       	addl	%ecx, %edx
100006ce8: 0f be 47 28                 	movsbl	40(%rdi), %eax
100006cec: 0f be 4e 28                 	movsbl	40(%rsi), %ecx
100006cf0: 0f af c8                    	imull	%eax, %ecx
100006cf3: 01 d1                       	addl	%edx, %ecx
100006cf5: 0f be 47 29                 	movsbl	41(%rdi), %eax
100006cf9: 0f be 56 29                 	movsbl	41(%rsi), %edx
100006cfd: 0f af d0                    	imull	%eax, %edx
100006d00: 01 ca                       	addl	%ecx, %edx
100006d02: 0f be 47 2a                 	movsbl	42(%rdi), %eax
100006d06: 0f be 4e 2a                 	movsbl	42(%rsi), %ecx
100006d0a: 0f af c8                    	imull	%eax, %ecx
100006d0d: 01 d1                       	addl	%edx, %ecx
100006d0f: 0f be 47 2b                 	movsbl	43(%rdi), %eax
100006d13: 0f be 56 2b                 	movsbl	43(%rsi), %edx
100006d17: 0f af d0                    	imull	%eax, %edx
100006d1a: 01 ca                       	addl	%ecx, %edx
100006d1c: 0f be 47 2c                 	movsbl	44(%rdi), %eax
100006d20: 0f be 4e 2c                 	movsbl	44(%rsi), %ecx
100006d24: 0f af c8                    	imull	%eax, %ecx
100006d27: 01 d1                       	addl	%edx, %ecx
100006d29: 0f be 47 2d                 	movsbl	45(%rdi), %eax
100006d2d: 0f be 56 2d                 	movsbl	45(%rsi), %edx
100006d31: 0f af d0                    	imull	%eax, %edx
100006d34: 01 ca                       	addl	%ecx, %edx
100006d36: 0f be 47 2e                 	movsbl	46(%rdi), %eax
100006d3a: 0f be 4e 2e                 	movsbl	46(%rsi), %ecx
100006d3e: 0f af c8                    	imull	%eax, %ecx
100006d41: 01 d1                       	addl	%edx, %ecx
100006d43: 0f be 57 2f                 	movsbl	47(%rdi), %edx
100006d47: 0f be 46 2f                 	movsbl	47(%rsi), %eax
100006d4b: 0f af c2                    	imull	%edx, %eax
100006d4e: 01 c8                       	addl	%ecx, %eax
100006d50: 5d                          	popq	%rbp
100006d51: c3                          	retq
100006d52: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100006d5c: 0f 1f 40 00                 	nopl	(%rax)
100006d60: 55                          	pushq	%rbp
100006d61: 48 89 e5                    	movq	%rsp, %rbp
100006d64: c4 e2 7d 21 47 08           	vpmovsxbd	8(%rdi), %ymm0
100006d6a: c4 e2 7d 21 4f 18           	vpmovsxbd	24(%rdi), %ymm1
100006d70: c4 e2 7d 21 17              	vpmovsxbd	(%rdi), %ymm2
100006d75: c4 e2 7d 21 5f 10           	vpmovsxbd	16(%rdi), %ymm3
100006d7b: c4 e2 7d 21 66 08           	vpmovsxbd	8(%rsi), %ymm4
100006d81: c4 e2 5d 40 c0              	vpmulld	%ymm0, %ymm4, %ymm0
100006d86: c4 e2 7d 21 66 18           	vpmovsxbd	24(%rsi), %ymm4
100006d8c: c4 e2 5d 40 c9              	vpmulld	%ymm1, %ymm4, %ymm1
100006d91: c4 e2 7d 21 26              	vpmovsxbd	(%rsi), %ymm4
100006d96: c4 e2 5d 40 d2              	vpmulld	%ymm2, %ymm4, %ymm2
100006d9b: c4 e2 7d 21 66 10           	vpmovsxbd	16(%rsi), %ymm4
100006da1: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006da5: c4 e2 5d 40 cb              	vpmulld	%ymm3, %ymm4, %ymm1
100006daa: c5 ed fe c9                 	vpaddd	%ymm1, %ymm2, %ymm1
100006dae: c5 f5 fe c0                 	vpaddd	%ymm0, %ymm1, %ymm0
100006db2: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006db8: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006dbc: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006dc1: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006dc5: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006dca: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006dce: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006dd2: 5d                          	popq	%rbp
100006dd3: c5 f8 77                    	vzeroupper
100006dd6: c3                          	retq

Disassembly of section __TEXT,__stubs:

0000000100006dd8 __stubs:
100006dd8: ff 25 22 32 00 00           	jmpq	*12834(%rip)
100006dde: ff 25 24 32 00 00           	jmpq	*12836(%rip)
100006de4: ff 25 26 32 00 00           	jmpq	*12838(%rip)
100006dea: ff 25 28 32 00 00           	jmpq	*12840(%rip)
100006df0: ff 25 2a 32 00 00           	jmpq	*12842(%rip)
100006df6: ff 25 2c 32 00 00           	jmpq	*12844(%rip)
100006dfc: ff 25 2e 32 00 00           	jmpq	*12846(%rip)
100006e02: ff 25 30 32 00 00           	jmpq	*12848(%rip)
100006e08: ff 25 32 32 00 00           	jmpq	*12850(%rip)
100006e0e: ff 25 34 32 00 00           	jmpq	*12852(%rip)
100006e14: ff 25 36 32 00 00           	jmpq	*12854(%rip)
100006e1a: ff 25 38 32 00 00           	jmpq	*12856(%rip)
100006e20: ff 25 3a 32 00 00           	jmpq	*12858(%rip)
100006e26: ff 25 3c 32 00 00           	jmpq	*12860(%rip)
100006e2c: ff 25 3e 32 00 00           	jmpq	*12862(%rip)
100006e32: ff 25 40 32 00 00           	jmpq	*12864(%rip)
100006e38: ff 25 42 32 00 00           	jmpq	*12866(%rip)
100006e3e: ff 25 44 32 00 00           	jmpq	*12868(%rip)
100006e44: ff 25 46 32 00 00           	jmpq	*12870(%rip)
100006e4a: ff 25 48 32 00 00           	jmpq	*12872(%rip)
100006e50: ff 25 4a 32 00 00           	jmpq	*12874(%rip)
100006e56: ff 25 4c 32 00 00           	jmpq	*12876(%rip)
100006e5c: ff 25 4e 32 00 00           	jmpq	*12878(%rip)
100006e62: ff 25 50 32 00 00           	jmpq	*12880(%rip)
100006e68: ff 25 52 32 00 00           	jmpq	*12882(%rip)
100006e6e: ff 25 54 32 00 00           	jmpq	*12884(%rip)
100006e74: ff 25 56 32 00 00           	jmpq	*12886(%rip)
100006e7a: ff 25 58 32 00 00           	jmpq	*12888(%rip)
100006e80: ff 25 5a 32 00 00           	jmpq	*12890(%rip)
100006e86: ff 25 5c 32 00 00           	jmpq	*12892(%rip)
100006e8c: ff 25 5e 32 00 00           	jmpq	*12894(%rip)
100006e92: ff 25 60 32 00 00           	jmpq	*12896(%rip)
100006e98: ff 25 62 32 00 00           	jmpq	*12898(%rip)
100006e9e: ff 25 64 32 00 00           	jmpq	*12900(%rip)
100006ea4: ff 25 66 32 00 00           	jmpq	*12902(%rip)
100006eaa: ff 25 68 32 00 00           	jmpq	*12904(%rip)
100006eb0: ff 25 6a 32 00 00           	jmpq	*12906(%rip)
100006eb6: ff 25 6c 32 00 00           	jmpq	*12908(%rip)
100006ebc: ff 25 6e 32 00 00           	jmpq	*12910(%rip)

Disassembly of section __TEXT,__stub_helper:

0000000100006ec4 __stub_helper:
100006ec4: 4c 8d 1d 6d 32 00 00        	leaq	12909(%rip), %r11
100006ecb: 41 53                       	pushq	%r11
100006ecd: ff 25 8d 21 00 00           	jmpq	*8589(%rip)
100006ed3: 90                          	nop
100006ed4: 68 7b 01 00 00              	pushq	$379
100006ed9: e9 e6 ff ff ff              	jmp	-26 <__stub_helper>
100006ede: 68 c9 02 00 00              	pushq	$713
100006ee3: e9 dc ff ff ff              	jmp	-36 <__stub_helper>
100006ee8: 68 44 00 00 00              	pushq	$68
100006eed: e9 d2 ff ff ff              	jmp	-46 <__stub_helper>
100006ef2: 68 a7 00 00 00              	pushq	$167
100006ef7: e9 c8 ff ff ff              	jmp	-56 <__stub_helper>
100006efc: 68 c8 00 00 00              	pushq	$200
100006f01: e9 be ff ff ff              	jmp	-66 <__stub_helper>
100006f06: 68 5b 03 00 00              	pushq	$859
100006f0b: e9 b4 ff ff ff              	jmp	-76 <__stub_helper>
100006f10: 68 e6 01 00 00              	pushq	$486
100006f15: e9 aa ff ff ff              	jmp	-86 <__stub_helper>
100006f1a: 68 34 02 00 00              	pushq	$564
100006f1f: e9 a0 ff ff ff              	jmp	-96 <__stub_helper>
100006f24: 68 e1 02 00 00              	pushq	$737
100006f29: e9 96 ff ff ff              	jmp	-106 <__stub_helper>
100006f2e: 68 17 00 00 00              	pushq	$23
100006f33: e9 8c ff ff ff              	jmp	-116 <__stub_helper>
100006f38: 68 f1 00 00 00              	pushq	$241
100006f3d: e9 82 ff ff ff              	jmp	-126 <__stub_helper>
100006f42: 68 12 01 00 00              	pushq	$274
100006f47: e9 78 ff ff ff              	jmp	-136 <__stub_helper>
100006f4c: 68 32 01 00 00              	pushq	$306
100006f51: e9 6e ff ff ff              	jmp	-146 <__stub_helper>
100006f56: 68 54 01 00 00              	pushq	$340
100006f5b: e9 64 ff ff ff              	jmp	-156 <__stub_helper>
100006f60: 68 23 03 00 00              	pushq	$803
100006f65: e9 5a ff ff ff              	jmp	-166 <__stub_helper>
100006f6a: 68 3e 03 00 00              	pushq	$830
100006f6f: e9 50 ff ff ff              	jmp	-176 <__stub_helper>
100006f74: 68 85 03 00 00              	pushq	$901
100006f79: e9 46 ff ff ff              	jmp	-186 <__stub_helper>
100006f7e: 68 b4 03 00 00              	pushq	$948
100006f83: e9 3c ff ff ff              	jmp	-196 <__stub_helper>
100006f88: 68 da 03 00 00              	pushq	$986
100006f8d: e9 32 ff ff ff              	jmp	-206 <__stub_helper>
100006f92: 68 2e 04 00 00              	pushq	$1070
100006f97: e9 28 ff ff ff              	jmp	-216 <__stub_helper>
100006f9c: 68 83 04 00 00              	pushq	$1155
100006fa1: e9 1e ff ff ff              	jmp	-226 <__stub_helper>
100006fa6: 68 d8 04 00 00              	pushq	$1240
100006fab: e9 14 ff ff ff              	jmp	-236 <__stub_helper>
100006fb0: 68 1f 05 00 00              	pushq	$1311
100006fb5: e9 0a ff ff ff              	jmp	-246 <__stub_helper>
100006fba: 68 63 05 00 00              	pushq	$1379
100006fbf: e9 00 ff ff ff              	jmp	-256 <__stub_helper>
100006fc4: 68 91 05 00 00              	pushq	$1425
100006fc9: e9 f6 fe ff ff              	jmp	-266 <__stub_helper>
100006fce: 68 af 05 00 00              	pushq	$1455
100006fd3: e9 ec fe ff ff              	jmp	-276 <__stub_helper>
100006fd8: 68 f0 05 00 00              	pushq	$1520
100006fdd: e9 e2 fe ff ff              	jmp	-286 <__stub_helper>
100006fe2: 68 14 06 00 00              	pushq	$1556
100006fe7: e9 d8 fe ff ff              	jmp	-296 <__stub_helper>
100006fec: 68 33 06 00 00              	pushq	$1587
100006ff1: e9 ce fe ff ff              	jmp	-306 <__stub_helper>
100006ff6: 68 52 06 00 00              	pushq	$1618
100006ffb: e9 c4 fe ff ff              	jmp	-316 <__stub_helper>
100007000: 68 6b 06 00 00              	pushq	$1643
100007005: e9 ba fe ff ff              	jmp	-326 <__stub_helper>
10000700a: 68 86 06 00 00              	pushq	$1670
10000700f: e9 b0 fe ff ff              	jmp	-336 <__stub_helper>
100007014: 68 00 00 00 00              	pushq	$0
100007019: e9 a6 fe ff ff              	jmp	-346 <__stub_helper>
10000701e: 68 9f 06 00 00              	pushq	$1695
100007023: e9 9c fe ff ff              	jmp	-356 <__stub_helper>
100007028: 68 b9 06 00 00              	pushq	$1721
10000702d: e9 92 fe ff ff              	jmp	-366 <__stub_helper>
100007032: 68 c9 06 00 00              	pushq	$1737
100007037: e9 88 fe ff ff              	jmp	-376 <__stub_helper>
