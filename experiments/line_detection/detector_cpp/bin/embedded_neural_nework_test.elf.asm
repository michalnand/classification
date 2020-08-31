
bin/embedded_neural_nework_test.elf:	file format Mach-O 64-bit x86-64


Disassembly of section __TEXT,__text:

00000001000026a0 __Z8get_timev:
1000026a0: 55                          	pushq	%rbp
1000026a1: 48 89 e5                    	movq	%rsp, %rbp
1000026a4: e8 bf 47 00 00              	callq	18367 <dyld_stub_binder+0x100006e68>
1000026a9: c4 e1 fb 2a c0              	vcvtsi2sd	%rax, %xmm0, %xmm0
1000026ae: c5 fb 5e 05 8a 49 00 00     	vdivsd	18826(%rip), %xmm0, %xmm0
1000026b6: 5d                          	popq	%rbp
1000026b7: c3                          	retq
1000026b8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

00000001000026c0 __Z14get_predictionRN2cv3MatER14ModelInterfacef:
1000026c0: 55                          	pushq	%rbp
1000026c1: 48 89 e5                    	movq	%rsp, %rbp
1000026c4: 41 57                       	pushq	%r15
1000026c6: 41 56                       	pushq	%r14
1000026c8: 41 55                       	pushq	%r13
1000026ca: 41 54                       	pushq	%r12
1000026cc: 53                          	pushq	%rbx
1000026cd: 48 81 ec 28 01 00 00        	subq	$296, %rsp
1000026d4: c5 fa 11 45 a8              	vmovss	%xmm0, -88(%rbp)
1000026d9: 49 89 d6                    	movq	%rdx, %r14
1000026dc: 49 89 f4                    	movq	%rsi, %r12
1000026df: 48 89 fb                    	movq	%rdi, %rbx
1000026e2: 48 8b 05 6f 69 00 00        	movq	26991(%rip), %rax
1000026e9: 48 8b 00                    	movq	(%rax), %rax
1000026ec: 48 89 45 d0                 	movq	%rax, -48(%rbp)
1000026f0: 8b 46 08                    	movl	8(%rsi), %eax
1000026f3: 8b 4e 0c                    	movl	12(%rsi), %ecx
1000026f6: c7 85 d0 fe ff ff 00 00 ff 42       	movl	$1124007936, -304(%rbp)
100002700: 48 8d 95 d8 fe ff ff        	leaq	-296(%rbp), %rdx
100002707: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000270b: c5 fc 11 85 d4 fe ff ff     	vmovups	%ymm0, -300(%rbp)
100002713: c5 fc 11 85 f0 fe ff ff     	vmovups	%ymm0, -272(%rbp)
10000271b: 48 89 95 10 ff ff ff        	movq	%rdx, -240(%rbp)
100002722: 48 8d 95 20 ff ff ff        	leaq	-224(%rbp), %rdx
100002729: 48 89 95 18 ff ff ff        	movq	%rdx, -232(%rbp)
100002730: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002734: c5 f8 11 85 20 ff ff ff     	vmovups	%xmm0, -224(%rbp)
10000273c: 89 4d b8                    	movl	%ecx, -72(%rbp)
10000273f: 89 45 bc                    	movl	%eax, -68(%rbp)
100002742: 4c 8d bd d0 fe ff ff        	leaq	-304(%rbp), %r15
100002749: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
10000274d: 4c 89 ff                    	movq	%r15, %rdi
100002750: be 02 00 00 00              	movl	$2, %esi
100002755: 31 c9                       	xorl	%ecx, %ecx
100002757: c5 f8 77                    	vzeroupper
10000275a: e8 9d 46 00 00              	callq	18077 <dyld_stub_binder+0x100006dfc>
10000275f: 48 c7 85 40 ff ff ff 00 00 00 00    	movq	$0, -192(%rbp)
10000276a: c7 85 30 ff ff ff 00 00 01 01       	movl	$16842752, -208(%rbp)
100002774: 4c 89 a5 38 ff ff ff        	movq	%r12, -200(%rbp)
10000277b: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
100002783: c7 45 b8 00 00 01 02        	movl	$33619968, -72(%rbp)
10000278a: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
10000278e: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002795: 48 8d 75 b8                 	leaq	-72(%rbp), %rsi
100002799: ba 06 00 00 00              	movl	$6, %edx
10000279e: 31 c9                       	xorl	%ecx, %ecx
1000027a0: e8 81 46 00 00              	callq	18049 <dyld_stub_binder+0x100006e26>
1000027a5: 41 8b 44 24 08              	movl	8(%r12), %eax
1000027aa: 41 8b 4c 24 0c              	movl	12(%r12), %ecx
1000027af: c7 85 30 ff ff ff 00 00 ff 42       	movl	$1124007936, -208(%rbp)
1000027b9: 48 8d 95 38 ff ff ff        	leaq	-200(%rbp), %rdx
1000027c0: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000027c4: c5 fc 11 85 34 ff ff ff     	vmovups	%ymm0, -204(%rbp)
1000027cc: c5 fc 11 85 50 ff ff ff     	vmovups	%ymm0, -176(%rbp)
1000027d4: 48 89 95 70 ff ff ff        	movq	%rdx, -144(%rbp)
1000027db: 48 8d 55 80                 	leaq	-128(%rbp), %rdx
1000027df: 48 89 95 78 ff ff ff        	movq	%rdx, -136(%rbp)
1000027e6: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000027ea: c5 f8 11 45 80              	vmovups	%xmm0, -128(%rbp)
1000027ef: 89 4d b8                    	movl	%ecx, -72(%rbp)
1000027f2: 89 45 bc                    	movl	%eax, -68(%rbp)
1000027f5: 4c 8d a5 30 ff ff ff        	leaq	-208(%rbp), %r12
1000027fc: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
100002800: 4c 89 e7                    	movq	%r12, %rdi
100002803: be 02 00 00 00              	movl	$2, %esi
100002808: 31 c9                       	xorl	%ecx, %ecx
10000280a: c5 f8 77                    	vzeroupper
10000280d: e8 ea 45 00 00              	callq	17898 <dyld_stub_binder+0x100006dfc>
100002812: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
10000281a: c7 45 b8 00 00 01 01        	movl	$16842752, -72(%rbp)
100002821: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
100002825: 48 c7 85 c0 fe ff ff 00 00 00 00    	movq	$0, -320(%rbp)
100002830: c7 85 b0 fe ff ff 00 00 01 02       	movl	$33619968, -336(%rbp)
10000283a: 4c 89 a5 b8 fe ff ff        	movq	%r12, -328(%rbp)
100002841: 41 8b 46 0c                 	movl	12(%r14), %eax
100002845: 41 8b 4e 10                 	movl	16(%r14), %ecx
100002849: 89 4d 90                    	movl	%ecx, -112(%rbp)
10000284c: 89 45 94                    	movl	%eax, -108(%rbp)
10000284f: 48 8d 7d b8                 	leaq	-72(%rbp), %rdi
100002853: 48 8d b5 b0 fe ff ff        	leaq	-336(%rbp), %rsi
10000285a: 48 8d 55 90                 	leaq	-112(%rbp), %rdx
10000285e: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002862: c5 f0 57 c9                 	vxorps	%xmm1, %xmm1, %xmm1
100002866: b9 03 00 00 00              	movl	$3, %ecx
10000286b: e8 a4 45 00 00              	callq	17828 <dyld_stub_binder+0x100006e14>
100002870: 41 8b 46 0c                 	movl	12(%r14), %eax
100002874: 85 c0                       	testl	%eax, %eax
100002876: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
10000287a: 4d 89 f7                    	movq	%r14, %r15
10000287d: 0f 84 7c 00 00 00           	je	124 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x23f>
100002883: 41 8b 4f 10                 	movl	16(%r15), %ecx
100002887: 31 d2                       	xorl	%edx, %edx
100002889: 45 31 e4                    	xorl	%r12d, %r12d
10000288c: 85 c9                       	testl	%ecx, %ecx
10000288e: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1dc>
100002890: 31 c9                       	xorl	%ecx, %ecx
100002892: ff c2                       	incl	%edx
100002894: 39 c2                       	cmpl	%eax, %edx
100002896: 73 67                       	jae	103 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x23f>
100002898: 85 c9                       	testl	%ecx, %ecx
10000289a: 74 f4                       	je	-12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1d0>
10000289c: 89 55 a0                    	movl	%edx, -96(%rbp)
10000289f: 4c 63 f2                    	movslq	%edx, %r14
1000028a2: 45 31 ed                    	xorl	%r13d, %r13d
1000028a5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000028af: 90                          	nop
1000028b0: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
1000028b7: 48 8b 00                    	movq	(%rax), %rax
1000028ba: 49 0f af c6                 	imulq	%r14, %rax
1000028be: 48 03 85 40 ff ff ff        	addq	-192(%rbp), %rax
1000028c5: 49 63 cd                    	movslq	%r13d, %rcx
1000028c8: 0f b6 1c 01                 	movzbl	(%rcx,%rax), %ebx
1000028cc: 4c 89 ff                    	movq	%r15, %rdi
1000028cf: e8 6c 21 00 00              	callq	8556 <__ZN14ModelInterface12input_bufferEv>
1000028d4: 43 8d 0c 2c                 	leal	(%r12,%r13), %ecx
1000028d8: d0 eb                       	shrb	%bl
1000028da: 89 c9                       	movl	%ecx, %ecx
1000028dc: 88 1c 08                    	movb	%bl, (%rax,%rcx)
1000028df: 41 ff c5                    	incl	%r13d
1000028e2: 41 8b 4f 10                 	movl	16(%r15), %ecx
1000028e6: 41 39 cd                    	cmpl	%ecx, %r13d
1000028e9: 72 c5                       	jb	-59 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1f0>
1000028eb: 41 8b 47 0c                 	movl	12(%r15), %eax
1000028ef: 45 01 ec                    	addl	%r13d, %r12d
1000028f2: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
1000028f6: 8b 55 a0                    	movl	-96(%rbp), %edx
1000028f9: ff c2                       	incl	%edx
1000028fb: 39 c2                       	cmpl	%eax, %edx
1000028fd: 72 99                       	jb	-103 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1d8>
1000028ff: 49 8b 07                    	movq	(%r15), %rax
100002902: 4c 89 ff                    	movq	%r15, %rdi
100002905: ff 50 10                    	callq	*16(%rax)
100002908: 41 8b 47 18                 	movl	24(%r15), %eax
10000290c: 41 8b 4f 1c                 	movl	28(%r15), %ecx
100002910: c7 03 00 00 ff 42           	movl	$1124007936, (%rbx)
100002916: 48 8d 53 08                 	leaq	8(%rbx), %rdx
10000291a: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000291e: c5 fc 11 43 04              	vmovups	%ymm0, 4(%rbx)
100002923: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
100002928: 48 89 53 40                 	movq	%rdx, 64(%rbx)
10000292c: 48 8d 53 50                 	leaq	80(%rbx), %rdx
100002930: 48 89 95 c8 fe ff ff        	movq	%rdx, -312(%rbp)
100002937: 48 89 53 48                 	movq	%rdx, 72(%rbx)
10000293b: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000293f: c5 f8 11 43 50              	vmovups	%xmm0, 80(%rbx)
100002944: 89 4d b8                    	movl	%ecx, -72(%rbp)
100002947: 89 45 bc                    	movl	%eax, -68(%rbp)
10000294a: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
10000294e: 48 89 df                    	movq	%rbx, %rdi
100002951: be 02 00 00 00              	movl	$2, %esi
100002956: 31 c9                       	xorl	%ecx, %ecx
100002958: c5 f8 77                    	vzeroupper
10000295b: e8 9c 44 00 00              	callq	17564 <dyld_stub_binder+0x100006dfc>
100002960: 41 8b 47 18                 	movl	24(%r15), %eax
100002964: 41 83 7f 14 01              	cmpl	$1, 20(%r15)
100002969: 4d 89 fc                    	movq	%r15, %r12
10000296c: 0f 85 c7 00 00 00           	jne	199 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x379>
100002972: 85 c0                       	testl	%eax, %eax
100002974: 0f 84 e2 01 00 00           	je	482 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
10000297a: c5 fa 10 45 a8              	vmovss	-88(%rbp), %xmm0
10000297f: c5 fa 59 05 01 47 00 00     	vmulss	18177(%rip), %xmm0, %xmm0
100002987: c5 fa 11 45 a0              	vmovss	%xmm0, -96(%rbp)
10000298c: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002991: 45 31 ff                    	xorl	%r15d, %r15d
100002994: 31 d2                       	xorl	%edx, %edx
100002996: 45 31 ed                    	xorl	%r13d, %r13d
100002999: 85 c9                       	testl	%ecx, %ecx
10000299b: 75 13                       	jne	19 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2f0>
10000299d: 0f 1f 00                    	nopl	(%rax)
1000029a0: 31 c9                       	xorl	%ecx, %ecx
1000029a2: ff c2                       	incl	%edx
1000029a4: 39 c2                       	cmpl	%eax, %edx
1000029a6: 0f 83 b0 01 00 00           	jae	432 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
1000029ac: 85 c9                       	testl	%ecx, %ecx
1000029ae: 74 f0                       	je	-16 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2e0>
1000029b0: 89 55 a8                    	movl	%edx, -88(%rbp)
1000029b3: 4c 63 f2                    	movslq	%edx, %r14
1000029b6: 31 db                       	xorl	%ebx, %ebx
1000029b8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
1000029c0: 4c 89 e7                    	movq	%r12, %rdi
1000029c3: e8 88 20 00 00              	callq	8328 <__ZN14ModelInterface13output_bufferEv>
1000029c8: 42 8d 0c 2b                 	leal	(%rbx,%r13), %ecx
1000029cc: 89 c9                       	movl	%ecx, %ecx
1000029ce: 0f b6 04 08                 	movzbl	(%rax,%rcx), %eax
1000029d2: 84 c0                       	testb	%al, %al
1000029d4: 41 0f 48 c7                 	cmovsl	%r15d, %eax
1000029d8: 0f be c8                    	movsbl	%al, %ecx
1000029db: c5 ea 2a c1                 	vcvtsi2ss	%ecx, %xmm2, %xmm0
1000029df: 48 8b 55 b0                 	movq	-80(%rbp), %rdx
1000029e3: 48 8b 4a 48                 	movq	72(%rdx), %rcx
1000029e7: 48 8b 09                    	movq	(%rcx), %rcx
1000029ea: 49 0f af ce                 	imulq	%r14, %rcx
1000029ee: 48 03 4a 10                 	addq	16(%rdx), %rcx
1000029f2: 48 63 db                    	movslq	%ebx, %rbx
1000029f5: 88 04 0b                    	movb	%al, (%rbx,%rcx)
1000029f8: 48 8b 42 48                 	movq	72(%rdx), %rax
1000029fc: 48 8b 00                    	movq	(%rax), %rax
1000029ff: 49 0f af c6                 	imulq	%r14, %rax
100002a03: 48 03 42 10                 	addq	16(%rdx), %rax
100002a07: c5 f8 2e 45 a0              	vucomiss	-96(%rbp), %xmm0
100002a0c: 0f 97 04 03                 	seta	(%rbx,%rax)
100002a10: ff c3                       	incl	%ebx
100002a12: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002a17: 39 cb                       	cmpl	%ecx, %ebx
100002a19: 72 a5                       	jb	-91 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x300>
100002a1b: 41 8b 44 24 18              	movl	24(%r12), %eax
100002a20: 41 01 dd                    	addl	%ebx, %r13d
100002a23: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002a27: 8b 55 a8                    	movl	-88(%rbp), %edx
100002a2a: ff c2                       	incl	%edx
100002a2c: 39 c2                       	cmpl	%eax, %edx
100002a2e: 0f 82 78 ff ff ff           	jb	-136 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2ec>
100002a34: e9 23 01 00 00              	jmp	291 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002a39: 85 c0                       	testl	%eax, %eax
100002a3b: 0f 84 1b 01 00 00           	je	283 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002a41: c5 fa 10 45 a8              	vmovss	-88(%rbp), %xmm0
100002a46: c5 fa 59 05 3a 46 00 00     	vmulss	17978(%rip), %xmm0, %xmm0
100002a4e: c5 fa 11 45 98              	vmovss	%xmm0, -104(%rbp)
100002a53: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002a58: 31 d2                       	xorl	%edx, %edx
100002a5a: 45 31 ff                    	xorl	%r15d, %r15d
100002a5d: 85 c9                       	testl	%ecx, %ecx
100002a5f: 75 29                       	jne	41 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3ca>
100002a61: e9 ea 00 00 00              	jmp	234 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x490>
100002a66: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002a70: 41 8b 44 24 18              	movl	24(%r12), %eax
100002a75: 8b 55 9c                    	movl	-100(%rbp), %edx
100002a78: ff c2                       	incl	%edx
100002a7a: 39 c2                       	cmpl	%eax, %edx
100002a7c: 0f 83 da 00 00 00           	jae	218 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002a82: 85 c9                       	testl	%ecx, %ecx
100002a84: 0f 84 c6 00 00 00           	je	198 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x490>
100002a8a: 89 55 9c                    	movl	%edx, -100(%rbp)
100002a8d: 48 63 c2                    	movslq	%edx, %rax
100002a90: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100002a94: 31 d2                       	xorl	%edx, %edx
100002a96: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100002a9a: 41 83 7c 24 14 00           	cmpl	$0, 20(%r12)
100002aa0: 75 60                       	jne	96 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x442>
100002aa2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002aac: 0f 1f 40 00                 	nopl	(%rax)
100002ab0: 41 b6 81                    	movb	$-127, %r14b
100002ab3: 45 31 ed                    	xorl	%r13d, %r13d
100002ab6: 41 0f be c6                 	movsbl	%r14b, %eax
100002aba: c5 ea 2a c0                 	vcvtsi2ss	%eax, %xmm2, %xmm0
100002abe: c5 f8 2e 45 98              	vucomiss	-104(%rbp), %xmm0
100002ac3: b8 00 00 00 00              	movl	$0, %eax
100002ac8: 44 0f 46 e8                 	cmovbel	%eax, %r13d
100002acc: 48 8b 43 48                 	movq	72(%rbx), %rax
100002ad0: 48 8b 00                    	movq	(%rax), %rax
100002ad3: 48 0f af 45 a8              	imulq	-88(%rbp), %rax
100002ad8: 48 03 43 10                 	addq	16(%rbx), %rax
100002adc: 48 8b 55 a0                 	movq	-96(%rbp), %rdx
100002ae0: 48 63 d2                    	movslq	%edx, %rdx
100002ae3: 44 88 2c 02                 	movb	%r13b, (%rdx,%rax)
100002ae7: ff c2                       	incl	%edx
100002ae9: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002aee: 39 ca                       	cmpl	%ecx, %edx
100002af0: 0f 83 7a ff ff ff           	jae	-134 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3b0>
100002af6: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100002afa: 41 83 7c 24 14 00           	cmpl	$0, 20(%r12)
100002b00: 74 ae                       	je	-82 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3f0>
100002b02: 41 b6 81                    	movb	$-127, %r14b
100002b05: 31 db                       	xorl	%ebx, %ebx
100002b07: 45 31 ed                    	xorl	%r13d, %r13d
100002b0a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100002b10: 4c 89 e7                    	movq	%r12, %rdi
100002b13: e8 38 1f 00 00              	callq	7992 <__ZN14ModelInterface13output_bufferEv>
100002b18: 41 8d 0c 1f                 	leal	(%r15,%rbx), %ecx
100002b1c: 89 c9                       	movl	%ecx, %ecx
100002b1e: 0f b6 04 08                 	movzbl	(%rax,%rcx), %eax
100002b22: 44 38 f0                    	cmpb	%r14b, %al
100002b25: 44 0f 4f eb                 	cmovgl	%ebx, %r13d
100002b29: 45 0f b6 f6                 	movzbl	%r14b, %r14d
100002b2d: 44 0f 4d f0                 	cmovgel	%eax, %r14d
100002b31: ff c3                       	incl	%ebx
100002b33: 41 3b 5c 24 14              	cmpl	20(%r12), %ebx
100002b38: 72 d6                       	jb	-42 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x450>
100002b3a: 41 01 df                    	addl	%ebx, %r15d
100002b3d: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002b41: e9 70 ff ff ff              	jmp	-144 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3f6>
100002b46: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002b50: 31 c9                       	xorl	%ecx, %ecx
100002b52: ff c2                       	incl	%edx
100002b54: 39 c2                       	cmpl	%eax, %edx
100002b56: 0f 82 26 ff ff ff           	jb	-218 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3c2>
100002b5c: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002b63: 48 85 c0                    	testq	%rax, %rax
100002b66: 74 12                       	je	18 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4ba>
100002b68: f0                          	lock
100002b69: ff 48 14                    	decl	20(%rax)
100002b6c: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4ba>
100002b6e: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002b75: e8 7c 42 00 00              	callq	17020 <dyld_stub_binder+0x100006df6>
100002b7a: 48 c7 85 68 ff ff ff 00 00 00 00    	movq	$0, -152(%rbp)
100002b85: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002b89: c5 fc 11 85 40 ff ff ff     	vmovups	%ymm0, -192(%rbp)
100002b91: 83 bd 34 ff ff ff 00        	cmpl	$0, -204(%rbp)
100002b98: 7e 2c                       	jle	44 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x506>
100002b9a: 48 8b 85 70 ff ff ff        	movq	-144(%rbp), %rax
100002ba1: 31 c9                       	xorl	%ecx, %ecx
100002ba3: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002bad: 0f 1f 00                    	nopl	(%rax)
100002bb0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002bb7: 48 ff c1                    	incq	%rcx
100002bba: 48 63 95 34 ff ff ff        	movslq	-204(%rbp), %rdx
100002bc1: 48 39 d1                    	cmpq	%rdx, %rcx
100002bc4: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4f0>
100002bc6: 48 8b bd 78 ff ff ff        	movq	-136(%rbp), %rdi
100002bcd: 48 8d 45 80                 	leaq	-128(%rbp), %rax
100002bd1: 48 39 c7                    	cmpq	%rax, %rdi
100002bd4: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x51e>
100002bd6: c5 f8 77                    	vzeroupper
100002bd9: e8 4e 42 00 00              	callq	16974 <dyld_stub_binder+0x100006e2c>
100002bde: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002be5: 48 85 c0                    	testq	%rax, %rax
100002be8: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x53f>
100002bea: f0                          	lock
100002beb: ff 48 14                    	decl	20(%rax)
100002bee: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x53f>
100002bf0: 48 8d bd d0 fe ff ff        	leaq	-304(%rbp), %rdi
100002bf7: c5 f8 77                    	vzeroupper
100002bfa: e8 f7 41 00 00              	callq	16887 <dyld_stub_binder+0x100006df6>
100002bff: 48 c7 85 08 ff ff ff 00 00 00 00    	movq	$0, -248(%rbp)
100002c0a: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002c0e: c5 fc 11 85 e0 fe ff ff     	vmovups	%ymm0, -288(%rbp)
100002c16: 83 bd d4 fe ff ff 00        	cmpl	$0, -300(%rbp)
100002c1d: 7e 27                       	jle	39 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x586>
100002c1f: 48 8b 85 10 ff ff ff        	movq	-240(%rbp), %rax
100002c26: 31 c9                       	xorl	%ecx, %ecx
100002c28: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002c30: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002c37: 48 ff c1                    	incq	%rcx
100002c3a: 48 63 95 d4 fe ff ff        	movslq	-300(%rbp), %rdx
100002c41: 48 39 d1                    	cmpq	%rdx, %rcx
100002c44: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x570>
100002c46: 48 8b bd 18 ff ff ff        	movq	-232(%rbp), %rdi
100002c4d: 48 8d 85 20 ff ff ff        	leaq	-224(%rbp), %rax
100002c54: 48 39 c7                    	cmpq	%rax, %rdi
100002c57: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5a1>
100002c59: c5 f8 77                    	vzeroupper
100002c5c: e8 cb 41 00 00              	callq	16843 <dyld_stub_binder+0x100006e2c>
100002c61: 48 8b 05 f0 63 00 00        	movq	25584(%rip), %rax
100002c68: 48 8b 00                    	movq	(%rax), %rax
100002c6b: 48 3b 45 d0                 	cmpq	-48(%rbp), %rax
100002c6f: 75 18                       	jne	24 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5c9>
100002c71: 48 89 d8                    	movq	%rbx, %rax
100002c74: 48 81 c4 28 01 00 00        	addq	$296, %rsp
100002c7b: 5b                          	popq	%rbx
100002c7c: 41 5c                       	popq	%r12
100002c7e: 41 5d                       	popq	%r13
100002c80: 41 5e                       	popq	%r14
100002c82: 41 5f                       	popq	%r15
100002c84: 5d                          	popq	%rbp
100002c85: c5 f8 77                    	vzeroupper
100002c88: c3                          	retq
100002c89: c5 f8 77                    	vzeroupper
100002c8c: e8 1f 42 00 00              	callq	16927 <dyld_stub_binder+0x100006eb0>
100002c91: 48 89 c7                    	movq	%rax, %rdi
100002c94: e8 27 17 00 00              	callq	5927 <_main+0x1550>
100002c99: 48 89 c7                    	movq	%rax, %rdi
100002c9c: e8 1f 17 00 00              	callq	5919 <_main+0x1550>
100002ca1: eb 1e                       	jmp	30 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x601>
100002ca3: eb 00                       	jmp	0 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5e5>
100002ca5: 49 89 c6                    	movq	%rax, %r14
100002ca8: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002caf: 48 85 c0                    	testq	%rax, %rax
100002cb2: 0f 85 0f 01 00 00           	jne	271 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x707>
100002cb8: e9 1f 01 00 00              	jmp	287 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002cbd: eb 02                       	jmp	2 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x601>
100002cbf: eb 14                       	jmp	20 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x615>
100002cc1: 49 89 c6                    	movq	%rax, %r14
100002cc4: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002ccb: 48 85 c0                    	testq	%rax, %rax
100002cce: 75 7f                       	jne	127 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x68f>
100002cd0: e9 8f 00 00 00              	jmp	143 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002cd5: 49 89 c6                    	movq	%rax, %r14
100002cd8: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002cdc: 48 8b 43 38                 	movq	56(%rbx), %rax
100002ce0: 48 85 c0                    	testq	%rax, %rax
100002ce3: 74 0e                       	je	14 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x633>
100002ce5: f0                          	lock
100002ce6: ff 48 14                    	decl	20(%rax)
100002ce9: 75 08                       	jne	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x633>
100002ceb: 48 89 df                    	movq	%rbx, %rdi
100002cee: e8 03 41 00 00              	callq	16643 <dyld_stub_binder+0x100006df6>
100002cf3: 48 c7 43 38 00 00 00 00     	movq	$0, 56(%rbx)
100002cfb: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002cff: c5 fc 11 43 10              	vmovups	%ymm0, 16(%rbx)
100002d04: 83 7b 04 00                 	cmpl	$0, 4(%rbx)
100002d08: 7e 20                       	jle	32 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x66a>
100002d0a: 48 8b 4d b0                 	movq	-80(%rbp), %rcx
100002d0e: 48 8d 41 04                 	leaq	4(%rcx), %rax
100002d12: 48 8b 49 40                 	movq	64(%rcx), %rcx
100002d16: 31 d2                       	xorl	%edx, %edx
100002d18: c7 04 91 00 00 00 00        	movl	$0, (%rcx,%rdx,4)
100002d1f: 48 ff c2                    	incq	%rdx
100002d22: 48 63 30                    	movslq	(%rax), %rsi
100002d25: 48 39 f2                    	cmpq	%rsi, %rdx
100002d28: 7c ee                       	jl	-18 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x658>
100002d2a: 48 8b 45 b0                 	movq	-80(%rbp), %rax
100002d2e: 48 8b 78 48                 	movq	72(%rax), %rdi
100002d32: 48 3b bd c8 fe ff ff        	cmpq	-312(%rbp), %rdi
100002d39: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x683>
100002d3b: c5 f8 77                    	vzeroupper
100002d3e: e8 e9 40 00 00              	callq	16617 <dyld_stub_binder+0x100006e2c>
100002d43: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002d4a: 48 85 c0                    	testq	%rax, %rax
100002d4d: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002d4f: f0                          	lock
100002d50: ff 48 14                    	decl	20(%rax)
100002d53: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002d55: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002d5c: c5 f8 77                    	vzeroupper
100002d5f: e8 92 40 00 00              	callq	16530 <dyld_stub_binder+0x100006df6>
100002d64: 48 c7 85 68 ff ff ff 00 00 00 00    	movq	$0, -152(%rbp)
100002d6f: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002d73: c5 fc 11 85 40 ff ff ff     	vmovups	%ymm0, -192(%rbp)
100002d7b: 83 bd 34 ff ff ff 00        	cmpl	$0, -204(%rbp)
100002d82: 7e 1f                       	jle	31 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6e3>
100002d84: 48 8b 85 70 ff ff ff        	movq	-144(%rbp), %rax
100002d8b: 31 c9                       	xorl	%ecx, %ecx
100002d8d: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002d94: 48 ff c1                    	incq	%rcx
100002d97: 48 63 95 34 ff ff ff        	movslq	-204(%rbp), %rdx
100002d9e: 48 39 d1                    	cmpq	%rdx, %rcx
100002da1: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6cd>
100002da3: 48 8b bd 78 ff ff ff        	movq	-136(%rbp), %rdi
100002daa: 48 8d 45 80                 	leaq	-128(%rbp), %rax
100002dae: 48 39 c7                    	cmpq	%rax, %rdi
100002db1: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6fb>
100002db3: c5 f8 77                    	vzeroupper
100002db6: e8 71 40 00 00              	callq	16497 <dyld_stub_binder+0x100006e2c>
100002dbb: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002dc2: 48 85 c0                    	testq	%rax, %rax
100002dc5: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002dc7: f0                          	lock
100002dc8: ff 48 14                    	decl	20(%rax)
100002dcb: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002dcd: 48 8d bd d0 fe ff ff        	leaq	-304(%rbp), %rdi
100002dd4: c5 f8 77                    	vzeroupper
100002dd7: e8 1a 40 00 00              	callq	16410 <dyld_stub_binder+0x100006df6>
100002ddc: 48 c7 85 08 ff ff ff 00 00 00 00    	movq	$0, -248(%rbp)
100002de7: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002deb: c5 fc 11 85 e0 fe ff ff     	vmovups	%ymm0, -288(%rbp)
100002df3: 83 bd d4 fe ff ff 00        	cmpl	$0, -300(%rbp)
100002dfa: 7e 2a                       	jle	42 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x766>
100002dfc: 48 8b 85 10 ff ff ff        	movq	-240(%rbp), %rax
100002e03: 31 c9                       	xorl	%ecx, %ecx
100002e05: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002e0f: 90                          	nop
100002e10: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002e17: 48 ff c1                    	incq	%rcx
100002e1a: 48 63 95 d4 fe ff ff        	movslq	-300(%rbp), %rdx
100002e21: 48 39 d1                    	cmpq	%rdx, %rcx
100002e24: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x750>
100002e26: 48 8b bd 18 ff ff ff        	movq	-232(%rbp), %rdi
100002e2d: 48 8d 85 20 ff ff ff        	leaq	-224(%rbp), %rax
100002e34: 48 39 c7                    	cmpq	%rax, %rdi
100002e37: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x781>
100002e39: c5 f8 77                    	vzeroupper
100002e3c: e8 eb 3f 00 00              	callq	16363 <dyld_stub_binder+0x100006e2c>
100002e41: 4c 89 f7                    	movq	%r14, %rdi
100002e44: c5 f8 77                    	vzeroupper
100002e47: e8 8c 3f 00 00              	callq	16268 <dyld_stub_binder+0x100006dd8>
100002e4c: 0f 0b                       	ud2
100002e4e: 48 89 c7                    	movq	%rax, %rdi
100002e51: e8 6a 15 00 00              	callq	5482 <_main+0x1550>
100002e56: 48 89 c7                    	movq	%rax, %rdi
100002e59: e8 62 15 00 00              	callq	5474 <_main+0x1550>
100002e5e: 48 89 c7                    	movq	%rax, %rdi
100002e61: e8 5a 15 00 00              	callq	5466 <_main+0x1550>
100002e66: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

0000000100002e70 _main:
100002e70: 55                          	pushq	%rbp
100002e71: 48 89 e5                    	movq	%rsp, %rbp
100002e74: 41 57                       	pushq	%r15
100002e76: 41 56                       	pushq	%r14
100002e78: 41 55                       	pushq	%r13
100002e7a: 41 54                       	pushq	%r12
100002e7c: 53                          	pushq	%rbx
100002e7d: 48 83 e4 e0                 	andq	$-32, %rsp
100002e81: 48 81 ec 00 04 00 00        	subq	$1024, %rsp
100002e88: 48 8b 05 c9 61 00 00        	movq	25033(%rip), %rax
100002e8f: 48 8b 00                    	movq	(%rax), %rax
100002e92: 48 89 84 24 e0 03 00 00     	movq	%rax, 992(%rsp)
100002e9a: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100002ea2: e8 f9 27 00 00              	callq	10233 <__ZN11LineNetworkC1Ev>
100002ea7: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002eab: c5 f9 7f 84 24 60 02 00 00  	vmovdqa	%xmm0, 608(%rsp)
100002eb4: 48 c7 84 24 70 02 00 00 00 00 00 00 	movq	$0, 624(%rsp)
100002ec0: bf 30 00 00 00              	movl	$48, %edi
100002ec5: e8 d4 3f 00 00              	callq	16340 <dyld_stub_binder+0x100006e9e>
100002eca: 48 89 84 24 70 02 00 00     	movq	%rax, 624(%rsp)
100002ed2: c5 f8 28 05 d6 41 00 00     	vmovaps	16854(%rip), %xmm0
100002eda: c5 f8 29 84 24 60 02 00 00  	vmovaps	%xmm0, 608(%rsp)
100002ee3: c5 fe 6f 05 cd 5f 00 00     	vmovdqu	24525(%rip), %ymm0
100002eeb: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002eef: 48 b9 69 64 65 6f 2e 6d 70 34       	movabsq	$3778640133568685161, %rcx
100002ef9: 48 89 48 20                 	movq	%rcx, 32(%rax)
100002efd: c6 40 28 00                 	movb	$0, 40(%rax)
100002f01: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100002f09: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
100002f11: 31 d2                       	xorl	%edx, %edx
100002f13: c5 f8 77                    	vzeroupper
100002f16: e8 c9 3e 00 00              	callq	16073 <dyld_stub_binder+0x100006de4>
100002f1b: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100002f23: 74 0d                       	je	13 <_main+0xc2>
100002f25: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
100002f2d: e8 60 3f 00 00              	callq	16224 <dyld_stub_binder+0x100006e92>
100002f32: 4c 8d 6c 24 68              	leaq	104(%rsp), %r13
100002f37: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002f3b: c5 f9 d6 44 24 78           	vmovq	%xmm0, 120(%rsp)
100002f41: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100002f49: 4c 8d b4 24 c0 03 00 00     	leaq	960(%rsp), %r14
100002f51: 4c 8d a4 24 c0 01 00 00     	leaq	448(%rsp), %r12
100002f59: eb 0e                       	jmp	14 <_main+0xf9>
100002f5b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100002f60: 45 85 ff                    	testl	%r15d, %r15d
100002f63: 0f 85 d1 0f 00 00           	jne	4049 <_main+0x10ca>
100002f69: 48 89 df                    	movq	%rbx, %rdi
100002f6c: c5 f8 77                    	vzeroupper
100002f6f: e8 c4 3e 00 00              	callq	16068 <dyld_stub_binder+0x100006e38>
100002f74: 84 c0                       	testb	%al, %al
100002f76: 0f 84 be 0f 00 00           	je	4030 <_main+0x10ca>
100002f7c: c7 44 24 18 00 00 ff 42     	movl	$1124007936, 24(%rsp)
100002f84: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002f88: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
100002f8d: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
100002f92: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002f96: 48 8d 44 24 20              	leaq	32(%rsp), %rax
100002f9b: 48 89 44 24 58              	movq	%rax, 88(%rsp)
100002fa0: 4c 89 6c 24 60              	movq	%r13, 96(%rsp)
100002fa5: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002fa9: c4 c1 7a 7f 45 00           	vmovdqu	%xmm0, (%r13)
100002faf: 48 89 df                    	movq	%rbx, %rdi
100002fb2: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100002fb7: c5 f8 77                    	vzeroupper
100002fba: e8 31 3e 00 00              	callq	15921 <dyld_stub_binder+0x100006df0>
100002fbf: 41 bf 03 00 00 00           	movl	$3, %r15d
100002fc5: 48 83 7c 24 28 00           	cmpq	$0, 40(%rsp)
100002fcb: 0f 84 8f 08 00 00           	je	2191 <_main+0x9f0>
100002fd1: 8b 44 24 1c                 	movl	28(%rsp), %eax
100002fd5: 83 f8 03                    	cmpl	$3, %eax
100002fd8: 0f 8d 62 03 00 00           	jge	866 <_main+0x4d0>
100002fde: 48 63 4c 24 20              	movslq	32(%rsp), %rcx
100002fe3: 48 63 74 24 24              	movslq	36(%rsp), %rsi
100002fe8: 48 0f af f1                 	imulq	%rcx, %rsi
100002fec: 85 c0                       	testl	%eax, %eax
100002fee: 0f 84 6c 08 00 00           	je	2156 <_main+0x9f0>
100002ff4: 48 85 f6                    	testq	%rsi, %rsi
100002ff7: 0f 84 63 08 00 00           	je	2147 <_main+0x9f0>
100002ffd: bf 19 00 00 00              	movl	$25, %edi
100003002: c5 f8 77                    	vzeroupper
100003005: e8 16 3e 00 00              	callq	15894 <dyld_stub_binder+0x100006e20>
10000300a: 3c 1b                       	cmpb	$27, %al
10000300c: 0f 84 4e 08 00 00           	je	2126 <_main+0x9f0>
100003012: e8 51 3e 00 00              	callq	15953 <dyld_stub_binder+0x100006e68>
100003017: 49 89 c5                    	movq	%rax, %r13
10000301a: 48 8d 9c 24 e0 00 00 00     	leaq	224(%rsp), %rbx
100003022: 48 89 df                    	movq	%rbx, %rdi
100003025: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
10000302a: 48 8d 94 24 08 02 00 00     	leaq	520(%rsp), %rdx
100003032: c5 f9 6e 05 52 40 00 00     	vmovd	16466(%rip), %xmm0
10000303a: e8 81 f6 ff ff              	callq	-2431 <__Z14get_predictionRN2cv3MatER14ModelInterfacef>
10000303f: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100003047: c5 fa 7e 05 01 40 00 00     	vmovq	16385(%rip), %xmm0
10000304f: 48 89 de                    	movq	%rbx, %rsi
100003052: e8 db 3d 00 00              	callq	15835 <dyld_stub_binder+0x100006e32>
100003057: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
10000305f: 48 85 c0                    	testq	%rax, %rax
100003062: 74 0e                       	je	14 <_main+0x202>
100003064: f0                          	lock
100003065: ff 48 14                    	decl	20(%rax)
100003068: 75 08                       	jne	8 <_main+0x202>
10000306a: 48 89 df                    	movq	%rbx, %rdi
10000306d: e8 84 3d 00 00              	callq	15748 <dyld_stub_binder+0x100006df6>
100003072: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
10000307e: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100003086: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000308a: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
10000308e: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100003096: 7e 2f                       	jle	47 <_main+0x257>
100003098: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
1000030a0: 31 c9                       	xorl	%ecx, %ecx
1000030a2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000030ac: 0f 1f 40 00                 	nopl	(%rax)
1000030b0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000030b7: 48 ff c1                    	incq	%rcx
1000030ba: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
1000030c2: 48 39 d1                    	cmpq	%rdx, %rcx
1000030c5: 7c e9                       	jl	-23 <_main+0x240>
1000030c7: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
1000030cf: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
1000030d7: 48 39 c7                    	cmpq	%rax, %rdi
1000030da: 74 08                       	je	8 <_main+0x274>
1000030dc: c5 f8 77                    	vzeroupper
1000030df: e8 48 3d 00 00              	callq	15688 <dyld_stub_binder+0x100006e2c>
1000030e4: c5 f8 77                    	vzeroupper
1000030e7: e8 7c 3d 00 00              	callq	15740 <dyld_stub_binder+0x100006e68>
1000030ec: 49 89 c7                    	movq	%rax, %r15
1000030ef: c7 84 24 e0 00 00 00 00 00 ff 42    	movl	$1124007936, 224(%rsp)
1000030fa: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100003102: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003106: c5 fe 7f 40 f4              	vmovdqu	%ymm0, -12(%rax)
10000310b: c5 fe 7f 40 10              	vmovdqu	%ymm0, 16(%rax)
100003110: 48 8b 44 24 20              	movq	32(%rsp), %rax
100003115: 48 8d 8c 24 e8 00 00 00     	leaq	232(%rsp), %rcx
10000311d: 48 89 8c 24 20 01 00 00     	movq	%rcx, 288(%rsp)
100003125: 48 8d 8c 24 30 01 00 00     	leaq	304(%rsp), %rcx
10000312d: 48 89 8c 24 28 01 00 00     	movq	%rcx, 296(%rsp)
100003135: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003139: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
10000313d: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
100003145: 48 89 df                    	movq	%rbx, %rdi
100003148: be 02 00 00 00              	movl	$2, %esi
10000314d: 4c 89 f2                    	movq	%r14, %rdx
100003150: 31 c9                       	xorl	%ecx, %ecx
100003152: c5 f8 77                    	vzeroupper
100003155: e8 a2 3c 00 00              	callq	15522 <dyld_stub_binder+0x100006dfc>
10000315a: 48 8d 9c 24 80 00 00 00     	leaq	128(%rsp), %rbx
100003162: 48 89 df                    	movq	%rbx, %rdi
100003165: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
10000316d: e8 6c 3c 00 00              	callq	15468 <dyld_stub_binder+0x100006dde>
100003172: 48 c7 84 24 70 01 00 00 00 00 00 00 	movq	$0, 368(%rsp)
10000317e: c7 84 24 60 01 00 00 00 00 01 02    	movl	$33619968, 352(%rsp)
100003189: 48 8d 84 24 e0 00 00 00     	leaq	224(%rsp), %rax
100003191: 48 89 84 24 68 01 00 00     	movq	%rax, 360(%rsp)
100003199: 8b 44 24 20                 	movl	32(%rsp), %eax
10000319d: 8b 4c 24 24                 	movl	36(%rsp), %ecx
1000031a1: 89 8c 24 b0 01 00 00        	movl	%ecx, 432(%rsp)
1000031a8: 89 84 24 b4 01 00 00        	movl	%eax, 436(%rsp)
1000031af: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000031b3: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
1000031b7: 48 89 df                    	movq	%rbx, %rdi
1000031ba: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
1000031c2: 48 8d 94 24 b0 01 00 00     	leaq	432(%rsp), %rdx
1000031ca: b9 01 00 00 00              	movl	$1, %ecx
1000031cf: e8 40 3c 00 00              	callq	15424 <dyld_stub_binder+0x100006e14>
1000031d4: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000031d8: c5 fd 7f 84 24 60 01 00 00  	vmovdqa	%ymm0, 352(%rsp)
1000031e1: c7 84 24 80 00 00 00 00 00 ff 42    	movl	$1124007936, 128(%rsp)
1000031ec: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
1000031f4: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
1000031f9: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
1000031fd: 48 8b 44 24 20              	movq	32(%rsp), %rax
100003202: 48 8d 8c 24 88 00 00 00     	leaq	136(%rsp), %rcx
10000320a: 48 89 8c 24 c0 00 00 00     	movq	%rcx, 192(%rsp)
100003212: 48 8d 8c 24 d0 00 00 00     	leaq	208(%rsp), %rcx
10000321a: 48 89 8c 24 c8 00 00 00     	movq	%rcx, 200(%rsp)
100003222: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003226: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
10000322a: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
100003232: 48 89 df                    	movq	%rbx, %rdi
100003235: be 02 00 00 00              	movl	$2, %esi
10000323a: 4c 89 f2                    	movq	%r14, %rdx
10000323d: b9 10 00 00 00              	movl	$16, %ecx
100003242: c5 f8 77                    	vzeroupper
100003245: e8 b2 3b 00 00              	callq	15282 <dyld_stub_binder+0x100006dfc>
10000324a: 48 89 df                    	movq	%rbx, %rdi
10000324d: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003255: e8 ae 3b 00 00              	callq	15278 <dyld_stub_binder+0x100006e08>
10000325a: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000325f: 48 85 c0                    	testq	%rax, %rax
100003262: 74 04                       	je	4 <_main+0x3f8>
100003264: f0                          	lock
100003265: ff 40 14                    	incl	20(%rax)
100003268: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003270: 48 85 c0                    	testq	%rax, %rax
100003273: 74 13                       	je	19 <_main+0x418>
100003275: f0                          	lock
100003276: ff 48 14                    	decl	20(%rax)
100003279: 75 0d                       	jne	13 <_main+0x418>
10000327b: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
100003283: e8 6e 3b 00 00              	callq	15214 <dyld_stub_binder+0x100006df6>
100003288: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
100003294: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
10000329c: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000032a0: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
1000032a5: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
1000032ad: 0f 8e 2c 06 00 00           	jle	1580 <_main+0xa6f>
1000032b3: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
1000032bb: 31 c9                       	xorl	%ecx, %ecx
1000032bd: 0f 1f 00                    	nopl	(%rax)
1000032c0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000032c7: 48 ff c1                    	incq	%rcx
1000032ca: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
1000032d2: 48 39 d1                    	cmpq	%rdx, %rcx
1000032d5: 7c e9                       	jl	-23 <_main+0x450>
1000032d7: 8b 44 24 18                 	movl	24(%rsp), %eax
1000032db: 89 84 24 80 00 00 00        	movl	%eax, 128(%rsp)
1000032e2: 83 fa 02                    	cmpl	$2, %edx
1000032e5: 0f 8f 0c 06 00 00           	jg	1548 <_main+0xa87>
1000032eb: 8b 44 24 1c                 	movl	28(%rsp), %eax
1000032ef: 83 f8 02                    	cmpl	$2, %eax
1000032f2: 0f 8f ff 05 00 00           	jg	1535 <_main+0xa87>
1000032f8: 89 84 24 84 00 00 00        	movl	%eax, 132(%rsp)
1000032ff: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100003303: 8b 44 24 24                 	movl	36(%rsp), %eax
100003307: 89 8c 24 88 00 00 00        	movl	%ecx, 136(%rsp)
10000330e: 89 84 24 8c 00 00 00        	movl	%eax, 140(%rsp)
100003315: 48 8b 44 24 60              	movq	96(%rsp), %rax
10000331a: 48 8b 10                    	movq	(%rax), %rdx
10000331d: 48 8b b4 24 c8 00 00 00     	movq	200(%rsp), %rsi
100003325: 48 89 16                    	movq	%rdx, (%rsi)
100003328: 48 8b 40 08                 	movq	8(%rax), %rax
10000332c: 48 89 46 08                 	movq	%rax, 8(%rsi)
100003330: e9 db 05 00 00              	jmp	1499 <_main+0xaa0>
100003335: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000333f: 90                          	nop
100003340: 48 8b 4c 24 58              	movq	88(%rsp), %rcx
100003345: 83 f8 0f                    	cmpl	$15, %eax
100003348: 77 0c                       	ja	12 <_main+0x4e6>
10000334a: be 01 00 00 00              	movl	$1, %esi
10000334f: 31 d2                       	xorl	%edx, %edx
100003351: e9 ea 04 00 00              	jmp	1258 <_main+0x9d0>
100003356: 89 c2                       	movl	%eax, %edx
100003358: 83 e2 f0                    	andl	$-16, %edx
10000335b: 48 8d 72 f0                 	leaq	-16(%rdx), %rsi
10000335f: 48 89 f7                    	movq	%rsi, %rdi
100003362: 48 c1 ef 04                 	shrq	$4, %rdi
100003366: 48 ff c7                    	incq	%rdi
100003369: 89 fb                       	movl	%edi, %ebx
10000336b: 83 e3 03                    	andl	$3, %ebx
10000336e: 48 83 fe 30                 	cmpq	$48, %rsi
100003372: 73 25                       	jae	37 <_main+0x529>
100003374: c4 e2 7d 59 05 cb 3c 00 00  	vpbroadcastq	15563(%rip), %ymm0
10000337d: 31 ff                       	xorl	%edi, %edi
10000337f: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
100003383: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100003387: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
10000338b: 48 85 db                    	testq	%rbx, %rbx
10000338e: 0f 85 0e 03 00 00           	jne	782 <_main+0x832>
100003394: e9 d0 03 00 00              	jmp	976 <_main+0x8f9>
100003399: 48 89 de                    	movq	%rbx, %rsi
10000339c: 48 29 fe                    	subq	%rdi, %rsi
10000339f: c4 e2 7d 59 05 a0 3c 00 00  	vpbroadcastq	15520(%rip), %ymm0
1000033a8: 31 ff                       	xorl	%edi, %edi
1000033aa: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
1000033ae: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
1000033b2: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
1000033b6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000033c0: c4 e2 7d 25 24 b9           	vpmovsxdq	(%rcx,%rdi,4), %ymm4
1000033c6: c4 e2 7d 25 6c b9 10        	vpmovsxdq	16(%rcx,%rdi,4), %ymm5
1000033cd: c4 e2 7d 25 74 b9 20        	vpmovsxdq	32(%rcx,%rdi,4), %ymm6
1000033d4: c4 e2 7d 25 7c b9 30        	vpmovsxdq	48(%rcx,%rdi,4), %ymm7
1000033db: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
1000033e0: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
1000033e4: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
1000033e9: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
1000033ee: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
1000033f3: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000033f9: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000033fd: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003402: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
100003407: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
10000340b: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
100003410: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
100003415: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
100003419: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000341e: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003422: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100003426: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
10000342b: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
10000342f: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100003434: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100003438: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000343c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003441: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003445: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003449: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
10000344e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
100003452: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
100003457: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
10000345b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000345f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003464: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003468: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000346c: c4 e2 7d 25 64 b9 40        	vpmovsxdq	64(%rcx,%rdi,4), %ymm4
100003473: c4 e2 7d 25 6c b9 50        	vpmovsxdq	80(%rcx,%rdi,4), %ymm5
10000347a: c4 e2 7d 25 74 b9 60        	vpmovsxdq	96(%rcx,%rdi,4), %ymm6
100003481: c4 e2 7d 25 7c b9 70        	vpmovsxdq	112(%rcx,%rdi,4), %ymm7
100003488: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
10000348d: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100003492: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100003497: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
10000349b: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
1000034a0: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000034a6: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000034aa: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000034af: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
1000034b4: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
1000034b8: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
1000034bd: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
1000034c1: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
1000034c6: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000034cb: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000034cf: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000034d3: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
1000034d8: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
1000034dc: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
1000034e1: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
1000034e5: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000034e9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000034ee: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
1000034f2: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000034f6: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
1000034fb: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
1000034ff: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100003504: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100003508: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
10000350c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003511: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003515: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100003519: c4 e2 7d 25 a4 b9 80 00 00 00       	vpmovsxdq	128(%rcx,%rdi,4), %ymm4
100003523: c4 e2 7d 25 ac b9 90 00 00 00       	vpmovsxdq	144(%rcx,%rdi,4), %ymm5
10000352d: c4 e2 7d 25 b4 b9 a0 00 00 00       	vpmovsxdq	160(%rcx,%rdi,4), %ymm6
100003537: c4 e2 7d 25 bc b9 b0 00 00 00       	vpmovsxdq	176(%rcx,%rdi,4), %ymm7
100003541: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
100003546: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
10000354b: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100003550: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
100003554: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100003559: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
10000355f: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100003563: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003568: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
10000356d: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100003571: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
100003576: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
10000357a: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
10000357f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003584: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003588: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
10000358c: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100003591: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100003595: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
10000359a: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
10000359e: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000035a2: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000035a7: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
1000035ab: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000035af: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
1000035b4: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
1000035b8: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
1000035bd: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
1000035c1: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000035c5: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000035ca: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
1000035ce: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000035d2: c4 e2 7d 25 a4 b9 c0 00 00 00       	vpmovsxdq	192(%rcx,%rdi,4), %ymm4
1000035dc: c4 e2 7d 25 ac b9 d0 00 00 00       	vpmovsxdq	208(%rcx,%rdi,4), %ymm5
1000035e6: c4 e2 7d 25 b4 b9 e0 00 00 00       	vpmovsxdq	224(%rcx,%rdi,4), %ymm6
1000035f0: c4 e2 7d 25 bc b9 f0 00 00 00       	vpmovsxdq	240(%rcx,%rdi,4), %ymm7
1000035fa: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
1000035ff: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100003604: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100003609: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
10000360d: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100003612: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003618: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000361c: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003621: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100003626: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
10000362a: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
10000362f: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100003633: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100003638: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000363d: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003641: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100003645: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
10000364a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
10000364e: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100003653: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100003657: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
10000365b: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003660: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003664: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003668: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
10000366d: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100003671: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100003676: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
10000367a: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
10000367e: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003683: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003687: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000368b: 48 83 c7 40                 	addq	$64, %rdi
10000368f: 48 83 c6 04                 	addq	$4, %rsi
100003693: 0f 85 27 fd ff ff           	jne	-729 <_main+0x550>
100003699: 48 85 db                    	testq	%rbx, %rbx
10000369c: 0f 84 c7 00 00 00           	je	199 <_main+0x8f9>
1000036a2: 48 8d 34 b9                 	leaq	(%rcx,%rdi,4), %rsi
1000036a6: 48 83 c6 30                 	addq	$48, %rsi
1000036aa: 48 c1 e3 06                 	shlq	$6, %rbx
1000036ae: 31 ff                       	xorl	%edi, %edi
1000036b0: c4 e2 7d 25 64 3e d0        	vpmovsxdq	-48(%rsi,%rdi), %ymm4
1000036b7: c4 e2 7d 25 6c 3e e0        	vpmovsxdq	-32(%rsi,%rdi), %ymm5
1000036be: c4 e2 7d 25 74 3e f0        	vpmovsxdq	-16(%rsi,%rdi), %ymm6
1000036c5: c4 e2 7d 25 3c 3e           	vpmovsxdq	(%rsi,%rdi), %ymm7
1000036cb: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
1000036d0: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
1000036d4: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
1000036d9: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
1000036de: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
1000036e3: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000036e9: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000036ed: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000036f2: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
1000036f7: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
1000036fb: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
100003700: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
100003705: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
100003709: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000370e: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003712: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100003716: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
10000371b: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
10000371f: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100003724: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100003728: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000372c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003731: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003735: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003739: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
10000373e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
100003742: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
100003747: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
10000374b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000374f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003754: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003758: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000375c: 48 83 c7 40                 	addq	$64, %rdi
100003760: 48 39 fb                    	cmpq	%rdi, %rbx
100003763: 0f 85 47 ff ff ff           	jne	-185 <_main+0x840>
100003769: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
10000376e: c5 dd f4 e0                 	vpmuludq	%ymm0, %ymm4, %ymm4
100003772: c5 d5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm5
100003777: c5 e5 f4 ed                 	vpmuludq	%ymm5, %ymm3, %ymm5
10000377b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000377f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003784: c5 e5 f4 c0                 	vpmuludq	%ymm0, %ymm3, %ymm0
100003788: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
10000378c: c5 e5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm3
100003791: c5 e5 f4 d8                 	vpmuludq	%ymm0, %ymm3, %ymm3
100003795: c5 dd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm4
10000379a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
10000379e: c5 dd d4 db                 	vpaddq	%ymm3, %ymm4, %ymm3
1000037a2: c5 e5 73 f3 20              	vpsllq	$32, %ymm3, %ymm3
1000037a7: c5 ed f4 c0                 	vpmuludq	%ymm0, %ymm2, %ymm0
1000037ab: c5 fd d4 c3                 	vpaddq	%ymm3, %ymm0, %ymm0
1000037af: c5 ed 73 d1 20              	vpsrlq	$32, %ymm1, %ymm2
1000037b4: c5 ed f4 d0                 	vpmuludq	%ymm0, %ymm2, %ymm2
1000037b8: c5 e5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm3
1000037bd: c5 f5 f4 db                 	vpmuludq	%ymm3, %ymm1, %ymm3
1000037c1: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
1000037c5: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
1000037ca: c5 f5 f4 c0                 	vpmuludq	%ymm0, %ymm1, %ymm0
1000037ce: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
1000037d2: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
1000037d8: c5 ed 73 d0 20              	vpsrlq	$32, %ymm0, %ymm2
1000037dd: c5 ed f4 d1                 	vpmuludq	%ymm1, %ymm2, %ymm2
1000037e1: c5 e5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm3
1000037e6: c5 fd f4 db                 	vpmuludq	%ymm3, %ymm0, %ymm3
1000037ea: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
1000037ee: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
1000037f3: c5 fd f4 c1                 	vpmuludq	%ymm1, %ymm0, %ymm0
1000037f7: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
1000037fb: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100003800: c5 e9 73 d0 20              	vpsrlq	$32, %xmm0, %xmm2
100003805: c5 e9 f4 d1                 	vpmuludq	%xmm1, %xmm2, %xmm2
100003809: c5 e1 73 d8 0c              	vpsrldq	$12, %xmm0, %xmm3
10000380e: c5 f9 f4 db                 	vpmuludq	%xmm3, %xmm0, %xmm3
100003812: c5 e1 d4 d2                 	vpaddq	%xmm2, %xmm3, %xmm2
100003816: c5 e9 73 f2 20              	vpsllq	$32, %xmm2, %xmm2
10000381b: c5 f9 f4 c1                 	vpmuludq	%xmm1, %xmm0, %xmm0
10000381f: c5 f9 d4 c2                 	vpaddq	%xmm2, %xmm0, %xmm0
100003823: c4 e1 f9 7e c6              	vmovq	%xmm0, %rsi
100003828: 48 39 c2                    	cmpq	%rax, %rdx
10000382b: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100003833: 74 1b                       	je	27 <_main+0x9e0>
100003835: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000383f: 90                          	nop
100003840: 48 63 3c 91                 	movslq	(%rcx,%rdx,4), %rdi
100003844: 48 0f af f7                 	imulq	%rdi, %rsi
100003848: 48 ff c2                    	incq	%rdx
10000384b: 48 39 d0                    	cmpq	%rdx, %rax
10000384e: 75 f0                       	jne	-16 <_main+0x9d0>
100003850: 85 c0                       	testl	%eax, %eax
100003852: 0f 85 9c f7 ff ff           	jne	-2148 <_main+0x184>
100003858: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100003860: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003865: 48 85 c0                    	testq	%rax, %rax
100003868: 74 13                       	je	19 <_main+0xa0d>
10000386a: f0                          	lock
10000386b: ff 48 14                    	decl	20(%rax)
10000386e: 75 0d                       	jne	13 <_main+0xa0d>
100003870: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
100003875: c5 f8 77                    	vzeroupper
100003878: e8 79 35 00 00              	callq	13689 <dyld_stub_binder+0x100006df6>
10000387d: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
100003886: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000388a: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
10000388f: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100003894: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
100003899: 7e 29                       	jle	41 <_main+0xa54>
10000389b: 48 8b 44 24 58              	movq	88(%rsp), %rax
1000038a0: 31 c9                       	xorl	%ecx, %ecx
1000038a2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000038ac: 0f 1f 40 00                 	nopl	(%rax)
1000038b0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000038b7: 48 ff c1                    	incq	%rcx
1000038ba: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
1000038bf: 48 39 d1                    	cmpq	%rdx, %rcx
1000038c2: 7c ec                       	jl	-20 <_main+0xa40>
1000038c4: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
1000038c9: 4c 39 ef                    	cmpq	%r13, %rdi
1000038cc: 0f 84 8e f6 ff ff           	je	-2418 <_main+0xf0>
1000038d2: c5 f8 77                    	vzeroupper
1000038d5: e8 52 35 00 00              	callq	13650 <dyld_stub_binder+0x100006e2c>
1000038da: e9 81 f6 ff ff              	jmp	-2431 <_main+0xf0>
1000038df: 8b 44 24 18                 	movl	24(%rsp), %eax
1000038e3: 89 84 24 80 00 00 00        	movl	%eax, 128(%rsp)
1000038ea: 8b 44 24 1c                 	movl	28(%rsp), %eax
1000038ee: 83 f8 02                    	cmpl	$2, %eax
1000038f1: 0f 8e 01 fa ff ff           	jle	-1535 <_main+0x488>
1000038f7: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
1000038ff: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100003904: c5 f8 77                    	vzeroupper
100003907: e8 f6 34 00 00              	callq	13558 <dyld_stub_binder+0x100006e02>
10000390c: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100003910: c4 c1 eb 2a c5              	vcvtsi2sd	%r13, %xmm2, %xmm0
100003915: c4 c1 eb 2a cf              	vcvtsi2sd	%r15, %xmm2, %xmm1
10000391a: c5 fb 10 15 1e 37 00 00     	vmovsd	14110(%rip), %xmm2
100003922: c5 fb 5e c2                 	vdivsd	%xmm2, %xmm0, %xmm0
100003926: c5 f3 5e ca                 	vdivsd	%xmm2, %xmm1, %xmm1
10000392a: c5 fc 10 54 24 28           	vmovups	40(%rsp), %ymm2
100003930: c5 fc 11 94 24 90 00 00 00  	vmovups	%ymm2, 144(%rsp)
100003939: c5 f9 10 54 24 48           	vmovupd	72(%rsp), %xmm2
10000393f: c5 f9 11 94 24 b0 00 00 00  	vmovupd	%xmm2, 176(%rsp)
100003948: 85 c9                       	testl	%ecx, %ecx
10000394a: 4d 89 f5                    	movq	%r14, %r13
10000394d: 0f 84 53 01 00 00           	je	339 <_main+0xc36>
100003953: 31 c0                       	xorl	%eax, %eax
100003955: 8b 74 24 24                 	movl	36(%rsp), %esi
100003959: 85 f6                       	testl	%esi, %esi
10000395b: be 00 00 00 00              	movl	$0, %esi
100003960: 75 21                       	jne	33 <_main+0xb13>
100003962: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000396c: 0f 1f 40 00                 	nopl	(%rax)
100003970: ff c0                       	incl	%eax
100003972: 39 c8                       	cmpl	%ecx, %eax
100003974: 0f 83 2c 01 00 00           	jae	300 <_main+0xc36>
10000397a: 85 f6                       	testl	%esi, %esi
10000397c: be 00 00 00 00              	movl	$0, %esi
100003981: 74 ed                       	je	-19 <_main+0xb00>
100003983: 48 63 c8                    	movslq	%eax, %rcx
100003986: 31 d2                       	xorl	%edx, %edx
100003988: c5 fb 10 25 c8 36 00 00     	vmovsd	14024(%rip), %xmm4
100003990: c5 fa 10 2d f8 36 00 00     	vmovss	14072(%rip), %xmm5
100003998: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
1000039a0: 48 8b 74 24 60              	movq	96(%rsp), %rsi
1000039a5: 48 8b 3e                    	movq	(%rsi), %rdi
1000039a8: 48 0f af f9                 	imulq	%rcx, %rdi
1000039ac: 48 03 7c 24 28              	addq	40(%rsp), %rdi
1000039b1: 48 63 d2                    	movslq	%edx, %rdx
1000039b4: 48 8d 34 52                 	leaq	(%rdx,%rdx,2), %rsi
1000039b8: 0f b6 3c 37                 	movzbl	(%rdi,%rsi), %edi
1000039bc: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
1000039c0: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
1000039c4: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
1000039c8: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
1000039d0: 48 8b 1b                    	movq	(%rbx), %rbx
1000039d3: 48 0f af d9                 	imulq	%rcx, %rbx
1000039d7: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
1000039df: 40 88 3c 33                 	movb	%dil, (%rbx,%rsi)
1000039e3: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
1000039e8: 48 8b 3f                    	movq	(%rdi), %rdi
1000039eb: 48 0f af f9                 	imulq	%rcx, %rdi
1000039ef: 48 03 7c 24 28              	addq	40(%rsp), %rdi
1000039f4: 0f b6 7c 37 01              	movzbl	1(%rdi,%rsi), %edi
1000039f9: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
1000039fd: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
100003a05: 48 8b 3f                    	movq	(%rdi), %rdi
100003a08: 48 0f af f9                 	imulq	%rcx, %rdi
100003a0c: 48 03 bc 24 f0 00 00 00     	addq	240(%rsp), %rdi
100003a14: 0f b6 3c 3a                 	movzbl	(%rdx,%rdi), %edi
100003a18: c5 ca 2a df                 	vcvtsi2ss	%edi, %xmm6, %xmm3
100003a1c: c5 e2 59 dd                 	vmulss	%xmm5, %xmm3, %xmm3
100003a20: c5 e2 5a db                 	vcvtss2sd	%xmm3, %xmm3, %xmm3
100003a24: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003a28: c5 eb 58 d3                 	vaddsd	%xmm3, %xmm2, %xmm2
100003a2c: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003a30: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100003a38: 48 8b 1b                    	movq	(%rbx), %rbx
100003a3b: 48 0f af d9                 	imulq	%rcx, %rbx
100003a3f: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100003a47: 40 88 7c 33 01              	movb	%dil, 1(%rbx,%rsi)
100003a4c: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003a51: 48 8b 3f                    	movq	(%rdi), %rdi
100003a54: 48 0f af f9                 	imulq	%rcx, %rdi
100003a58: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100003a5d: 0f b6 7c 37 02              	movzbl	2(%rdi,%rsi), %edi
100003a62: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003a66: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003a6a: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003a6e: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100003a76: 48 8b 1b                    	movq	(%rbx), %rbx
100003a79: 48 0f af d9                 	imulq	%rcx, %rbx
100003a7d: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100003a85: 40 88 7c 33 02              	movb	%dil, 2(%rbx,%rsi)
100003a8a: ff c2                       	incl	%edx
100003a8c: 8b 74 24 24                 	movl	36(%rsp), %esi
100003a90: 39 f2                       	cmpl	%esi, %edx
100003a92: 0f 82 08 ff ff ff           	jb	-248 <_main+0xb30>
100003a98: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100003a9c: ff c0                       	incl	%eax
100003a9e: 39 c8                       	cmpl	%ecx, %eax
100003aa0: 0f 82 d4 fe ff ff           	jb	-300 <_main+0xb0a>
100003aa6: c5 fb 10 15 b2 35 00 00     	vmovsd	13746(%rip), %xmm2
100003aae: c5 eb 59 54 24 78           	vmulsd	120(%rsp), %xmm2, %xmm2
100003ab4: c5 f3 5c c0                 	vsubsd	%xmm0, %xmm1, %xmm0
100003ab8: c5 fb 58 05 a8 35 00 00     	vaddsd	13736(%rip), %xmm0, %xmm0
100003ac0: c5 fb 10 0d a8 35 00 00     	vmovsd	13736(%rip), %xmm1
100003ac8: c5 f3 5e c0                 	vdivsd	%xmm0, %xmm1, %xmm0
100003acc: c5 eb 58 c0                 	vaddsd	%xmm0, %xmm2, %xmm0
100003ad0: 8b 9c 24 28 02 00 00        	movl	552(%rsp), %ebx
100003ad7: c5 fb 11 44 24 78           	vmovsd	%xmm0, 120(%rsp)
100003add: c5 f8 77                    	vzeroupper
100003ae0: e8 d7 33 00 00              	callq	13271 <dyld_stub_binder+0x100006ebc>
100003ae5: c5 fb 2c f0                 	vcvttsd2si	%xmm0, %esi
100003ae9: 4c 89 e7                    	movq	%r12, %rdi
100003aec: e8 95 33 00 00              	callq	13205 <dyld_stub_binder+0x100006e86>
100003af1: 4c 89 e7                    	movq	%r12, %rdi
100003af4: 31 f6                       	xorl	%esi, %esi
100003af6: 48 8d 15 e4 53 00 00        	leaq	21476(%rip), %rdx
100003afd: e8 54 33 00 00              	callq	13140 <dyld_stub_binder+0x100006e56>
100003b02: 48 8b 48 10                 	movq	16(%rax), %rcx
100003b06: 48 89 8c 24 50 01 00 00     	movq	%rcx, 336(%rsp)
100003b0e: c5 f9 10 00                 	vmovupd	(%rax), %xmm0
100003b12: c5 f9 29 84 24 40 01 00 00  	vmovapd	%xmm0, 320(%rsp)
100003b1b: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003b1f: c5 f9 11 00                 	vmovupd	%xmm0, (%rax)
100003b23: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003b2b: 48 8d bc 24 40 01 00 00     	leaq	320(%rsp), %rdi
100003b33: 48 8d 35 ae 53 00 00        	leaq	21422(%rip), %rsi
100003b3a: e8 0b 33 00 00              	callq	13067 <dyld_stub_binder+0x100006e4a>
100003b3f: c4 e1 cb 2a c3              	vcvtsi2sd	%rbx, %xmm6, %xmm0
100003b44: c5 fb 59 44 24 78           	vmulsd	120(%rsp), %xmm0, %xmm0
100003b4a: c5 fb 5e 05 26 35 00 00     	vdivsd	13606(%rip), %xmm0, %xmm0
100003b52: 48 8b 48 10                 	movq	16(%rax), %rcx
100003b56: 48 89 8c 24 d0 03 00 00     	movq	%rcx, 976(%rsp)
100003b5e: c5 f9 10 08                 	vmovupd	(%rax), %xmm1
100003b62: c5 f9 29 8c 24 c0 03 00 00  	vmovapd	%xmm1, 960(%rsp)
100003b6b: c5 f1 57 c9                 	vxorpd	%xmm1, %xmm1, %xmm1
100003b6f: c5 f9 11 08                 	vmovupd	%xmm1, (%rax)
100003b73: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003b7b: 48 8d bc 24 98 01 00 00     	leaq	408(%rsp), %rdi
100003b83: e8 f8 32 00 00              	callq	13048 <dyld_stub_binder+0x100006e80>
100003b88: 0f b6 94 24 98 01 00 00     	movzbl	408(%rsp), %edx
100003b90: f6 c2 01                    	testb	$1, %dl
100003b93: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100003b9b: 74 12                       	je	18 <_main+0xd3f>
100003b9d: 48 8b b4 24 a8 01 00 00     	movq	424(%rsp), %rsi
100003ba5: 48 8b 94 24 a0 01 00 00     	movq	416(%rsp), %rdx
100003bad: eb 0b                       	jmp	11 <_main+0xd4a>
100003baf: 48 d1 ea                    	shrq	%rdx
100003bb2: 48 8d b4 24 99 01 00 00     	leaq	409(%rsp), %rsi
100003bba: 4c 89 ef                    	movq	%r13, %rdi
100003bbd: e8 8e 32 00 00              	callq	12942 <dyld_stub_binder+0x100006e50>
100003bc2: 48 8b 48 10                 	movq	16(%rax), %rcx
100003bc6: 48 89 8c 24 70 01 00 00     	movq	%rcx, 368(%rsp)
100003bce: c5 f8 10 00                 	vmovups	(%rax), %xmm0
100003bd2: c5 f8 29 84 24 60 01 00 00  	vmovaps	%xmm0, 352(%rsp)
100003bdb: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003bdf: c5 f8 11 00                 	vmovups	%xmm0, (%rax)
100003be3: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003beb: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
100003bf3: 0f 85 80 01 00 00           	jne	384 <_main+0xf09>
100003bf9: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003c01: 0f 85 8d 01 00 00           	jne	397 <_main+0xf24>
100003c07: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003c0f: 0f 85 9a 01 00 00           	jne	410 <_main+0xf3f>
100003c15: 4d 89 e7                    	movq	%r12, %r15
100003c18: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003c20: 74 0d                       	je	13 <_main+0xdbf>
100003c22: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
100003c2a: e8 63 32 00 00              	callq	12899 <dyld_stub_binder+0x100006e92>
100003c2f: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
100003c3b: c7 84 24 c0 03 00 00 00 00 01 03    	movl	$50397184, 960(%rsp)
100003c46: 4c 8d a4 24 80 00 00 00     	leaq	128(%rsp), %r12
100003c4e: 4c 89 a4 24 c8 03 00 00     	movq	%r12, 968(%rsp)
100003c56: 48 b8 1e 00 00 00 1e 00 00 00       	movabsq	$128849018910, %rax
100003c60: 48 89 84 24 b8 01 00 00     	movq	%rax, 440(%rsp)
100003c68: c5 fc 28 05 50 34 00 00     	vmovaps	13392(%rip), %ymm0
100003c70: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100003c79: c7 44 24 08 00 00 00 00     	movl	$0, 8(%rsp)
100003c81: c7 04 24 10 00 00 00        	movl	$16, (%rsp)
100003c88: 4c 89 ef                    	movq	%r13, %rdi
100003c8b: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003c93: 48 8d 94 24 b8 01 00 00     	leaq	440(%rsp), %rdx
100003c9b: 31 c9                       	xorl	%ecx, %ecx
100003c9d: c5 fb 10 05 db 33 00 00     	vmovsd	13275(%rip), %xmm0
100003ca5: 4c 8d 84 24 40 02 00 00     	leaq	576(%rsp), %r8
100003cad: 41 b9 02 00 00 00           	movl	$2, %r9d
100003cb3: c5 f8 77                    	vzeroupper
100003cb6: e8 5f 31 00 00              	callq	12639 <dyld_stub_binder+0x100006e1a>
100003cbb: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003cbf: c5 f9 29 84 24 c0 03 00 00  	vmovapd	%xmm0, 960(%rsp)
100003cc8: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
100003cd4: c6 84 24 c0 03 00 00 0a     	movb	$10, 960(%rsp)
100003cdc: 48 8d 84 24 c1 03 00 00     	leaq	961(%rsp), %rax
100003ce4: c6 40 04 65                 	movb	$101, 4(%rax)
100003ce8: c7 00 66 72 61 6d           	movl	$1835102822, (%rax)
100003cee: c6 84 24 c6 03 00 00 00     	movb	$0, 966(%rsp)
100003cf6: 48 c7 84 24 50 01 00 00 00 00 00 00 	movq	$0, 336(%rsp)
100003d02: c7 84 24 40 01 00 00 00 00 01 01    	movl	$16842752, 320(%rsp)
100003d0d: 4c 89 a4 24 48 01 00 00     	movq	%r12, 328(%rsp)
100003d15: 4c 89 ef                    	movq	%r13, %rdi
100003d18: 48 8d b4 24 40 01 00 00     	leaq	320(%rsp), %rsi
100003d20: e8 e9 30 00 00              	callq	12521 <dyld_stub_binder+0x100006e0e>
100003d25: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003d2d: 4d 89 fc                    	movq	%r15, %r12
100003d30: 4c 8d 6c 24 68              	leaq	104(%rsp), %r13
100003d35: 0f 85 97 00 00 00           	jne	151 <_main+0xf62>
100003d3b: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003d43: 4c 8d bc 24 80 00 00 00     	leaq	128(%rsp), %r15
100003d4b: 0f 85 a4 00 00 00           	jne	164 <_main+0xf85>
100003d51: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003d59: 48 85 c0                    	testq	%rax, %rax
100003d5c: 0f 84 b1 00 00 00           	je	177 <_main+0xfa3>
100003d62: f0                          	lock
100003d63: ff 48 14                    	decl	20(%rax)
100003d66: 0f 85 a7 00 00 00           	jne	167 <_main+0xfa3>
100003d6c: 4c 89 ff                    	movq	%r15, %rdi
100003d6f: e8 82 30 00 00              	callq	12418 <dyld_stub_binder+0x100006df6>
100003d74: e9 9a 00 00 00              	jmp	154 <_main+0xfa3>
100003d79: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100003d81: e8 0c 31 00 00              	callq	12556 <dyld_stub_binder+0x100006e92>
100003d86: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003d8e: 0f 84 73 fe ff ff           	je	-397 <_main+0xd97>
100003d94: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003d9c: e8 f1 30 00 00              	callq	12529 <dyld_stub_binder+0x100006e92>
100003da1: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003da9: 0f 84 66 fe ff ff           	je	-410 <_main+0xda5>
100003daf: 48 8b bc 24 50 01 00 00     	movq	336(%rsp), %rdi
100003db7: e8 d6 30 00 00              	callq	12502 <dyld_stub_binder+0x100006e92>
100003dbc: 4d 89 e7                    	movq	%r12, %r15
100003dbf: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003dc7: 0f 85 55 fe ff ff           	jne	-427 <_main+0xdb2>
100003dcd: e9 5d fe ff ff              	jmp	-419 <_main+0xdbf>
100003dd2: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003dda: e8 b3 30 00 00              	callq	12467 <dyld_stub_binder+0x100006e92>
100003ddf: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003de7: 4c 8d bc 24 80 00 00 00     	leaq	128(%rsp), %r15
100003def: 0f 84 5c ff ff ff           	je	-164 <_main+0xee1>
100003df5: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
100003dfd: e8 90 30 00 00              	callq	12432 <dyld_stub_binder+0x100006e92>
100003e02: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003e0a: 48 85 c0                    	testq	%rax, %rax
100003e0d: 0f 85 4f ff ff ff           	jne	-177 <_main+0xef2>
100003e13: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
100003e1f: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
100003e27: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003e2b: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100003e30: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
100003e38: 7e 2d                       	jle	45 <_main+0xff7>
100003e3a: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
100003e42: 31 c9                       	xorl	%ecx, %ecx
100003e44: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003e4e: 66 90                       	nop
100003e50: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003e57: 48 ff c1                    	incq	%rcx
100003e5a: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
100003e62: 48 39 d1                    	cmpq	%rdx, %rcx
100003e65: 7c e9                       	jl	-23 <_main+0xfe0>
100003e67: 48 8b bc 24 c8 00 00 00     	movq	200(%rsp), %rdi
100003e6f: 48 8d 84 24 d0 00 00 00     	leaq	208(%rsp), %rax
100003e77: 48 39 c7                    	cmpq	%rax, %rdi
100003e7a: 74 08                       	je	8 <_main+0x1014>
100003e7c: c5 f8 77                    	vzeroupper
100003e7f: e8 a8 2f 00 00              	callq	12200 <dyld_stub_binder+0x100006e2c>
100003e84: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
100003e8c: 48 85 c0                    	testq	%rax, %rax
100003e8f: 74 16                       	je	22 <_main+0x1037>
100003e91: f0                          	lock
100003e92: ff 48 14                    	decl	20(%rax)
100003e95: 75 10                       	jne	16 <_main+0x1037>
100003e97: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003e9f: c5 f8 77                    	vzeroupper
100003ea2: e8 4f 2f 00 00              	callq	12111 <dyld_stub_binder+0x100006df6>
100003ea7: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100003eb3: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100003ebb: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003ebf: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
100003ec3: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100003ecb: 7e 2a                       	jle	42 <_main+0x1087>
100003ecd: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100003ed5: 31 c9                       	xorl	%ecx, %ecx
100003ed7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100003ee0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003ee7: 48 ff c1                    	incq	%rcx
100003eea: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
100003ef2: 48 39 d1                    	cmpq	%rdx, %rcx
100003ef5: 7c e9                       	jl	-23 <_main+0x1070>
100003ef7: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
100003eff: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
100003f07: 48 39 c7                    	cmpq	%rax, %rdi
100003f0a: 74 08                       	je	8 <_main+0x10a4>
100003f0c: c5 f8 77                    	vzeroupper
100003f0f: e8 18 2f 00 00              	callq	12056 <dyld_stub_binder+0x100006e2c>
100003f14: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100003f1c: c5 f8 77                    	vzeroupper
100003f1f: e8 ac 04 00 00              	callq	1196 <_main+0x1560>
100003f24: 45 31 ff                    	xorl	%r15d, %r15d
100003f27: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003f2c: 48 85 c0                    	testq	%rax, %rax
100003f2f: 0f 85 35 f9 ff ff           	jne	-1739 <_main+0x9fa>
100003f35: e9 43 f9 ff ff              	jmp	-1725 <_main+0xa0d>
100003f3a: 48 8b 3d ff 50 00 00        	movq	20735(%rip), %rdi
100003f41: 48 8d 35 b4 4f 00 00        	leaq	20404(%rip), %rsi
100003f48: ba 0d 00 00 00              	movl	$13, %edx
100003f4d: c5 f8 77                    	vzeroupper
100003f50: e8 0b 06 00 00              	callq	1547 <_main+0x16f0>
100003f55: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100003f5d: e8 88 2e 00 00              	callq	11912 <dyld_stub_binder+0x100006dea>
100003f62: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100003f6a: e8 01 0a 00 00              	callq	2561 <__ZN14ModelInterfaceD2Ev>
100003f6f: 48 8b 05 e2 50 00 00        	movq	20706(%rip), %rax
100003f76: 48 8b 00                    	movq	(%rax), %rax
100003f79: 48 3b 84 24 e0 03 00 00     	cmpq	992(%rsp), %rax
100003f81: 75 11                       	jne	17 <_main+0x1124>
100003f83: 31 c0                       	xorl	%eax, %eax
100003f85: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100003f89: 5b                          	popq	%rbx
100003f8a: 41 5c                       	popq	%r12
100003f8c: 41 5d                       	popq	%r13
100003f8e: 41 5e                       	popq	%r14
100003f90: 41 5f                       	popq	%r15
100003f92: 5d                          	popq	%rbp
100003f93: c3                          	retq
100003f94: e8 17 2f 00 00              	callq	12055 <dyld_stub_binder+0x100006eb0>
100003f99: e9 f7 03 00 00              	jmp	1015 <_main+0x1525>
100003f9e: 48 89 c3                    	movq	%rax, %rbx
100003fa1: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100003fa9: 0f 84 f9 03 00 00           	je	1017 <_main+0x1538>
100003faf: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
100003fb7: e8 d6 2e 00 00              	callq	11990 <dyld_stub_binder+0x100006e92>
100003fbc: e9 e7 03 00 00              	jmp	999 <_main+0x1538>
100003fc1: 48 89 c3                    	movq	%rax, %rbx
100003fc4: e9 df 03 00 00              	jmp	991 <_main+0x1538>
100003fc9: 48 89 c7                    	movq	%rax, %rdi
100003fcc: e8 ef 03 00 00              	callq	1007 <_main+0x1550>
100003fd1: 48 89 c7                    	movq	%rax, %rdi
100003fd4: e8 e7 03 00 00              	callq	999 <_main+0x1550>
100003fd9: 48 89 c7                    	movq	%rax, %rdi
100003fdc: e8 df 03 00 00              	callq	991 <_main+0x1550>
100003fe1: 48 89 c3                    	movq	%rax, %rbx
100003fe4: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003fec: 48 85 c0                    	testq	%rax, %rax
100003fef: 0f 85 c8 01 00 00           	jne	456 <_main+0x134d>
100003ff5: e9 d6 01 00 00              	jmp	470 <_main+0x1360>
100003ffa: 48 89 c3                    	movq	%rax, %rbx
100003ffd: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
100004005: 48 85 c0                    	testq	%rax, %rax
100004008: 74 13                       	je	19 <_main+0x11ad>
10000400a: f0                          	lock
10000400b: ff 48 14                    	decl	20(%rax)
10000400e: 75 0d                       	jne	13 <_main+0x11ad>
100004010: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100004018: e8 d9 2d 00 00              	callq	11737 <dyld_stub_binder+0x100006df6>
10000401d: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100004029: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000402d: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100004035: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100004039: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100004041: 7e 21                       	jle	33 <_main+0x11f4>
100004043: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
10000404b: 31 c9                       	xorl	%ecx, %ecx
10000404d: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004054: 48 ff c1                    	incq	%rcx
100004057: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
10000405f: 48 39 d1                    	cmpq	%rdx, %rcx
100004062: 7c e9                       	jl	-23 <_main+0x11dd>
100004064: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
10000406c: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
100004074: 48 39 c7                    	cmpq	%rax, %rdi
100004077: 0f 84 96 02 00 00           	je	662 <_main+0x14a3>
10000407d: c5 f8 77                    	vzeroupper
100004080: e8 a7 2d 00 00              	callq	11687 <dyld_stub_binder+0x100006e2c>
100004085: e9 89 02 00 00              	jmp	649 <_main+0x14a3>
10000408a: 48 89 c7                    	movq	%rax, %rdi
10000408d: e8 2e 03 00 00              	callq	814 <_main+0x1550>
100004092: 48 89 c3                    	movq	%rax, %rbx
100004095: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000409a: 48 85 c0                    	testq	%rax, %rax
10000409d: 0f 85 7a 02 00 00           	jne	634 <_main+0x14ad>
1000040a3: e9 88 02 00 00              	jmp	648 <_main+0x14c0>
1000040a8: 48 89 c3                    	movq	%rax, %rbx
1000040ab: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000040b3: 74 1f                       	je	31 <_main+0x1264>
1000040b5: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
1000040bd: e8 d0 2d 00 00              	callq	11728 <dyld_stub_binder+0x100006e92>
1000040c2: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
1000040ca: 75 16                       	jne	22 <_main+0x1272>
1000040cc: e9 df 00 00 00              	jmp	223 <_main+0x1340>
1000040d1: 48 89 c3                    	movq	%rax, %rbx
1000040d4: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
1000040dc: 0f 84 ce 00 00 00           	je	206 <_main+0x1340>
1000040e2: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
1000040ea: e9 aa 00 00 00              	jmp	170 <_main+0x1329>
1000040ef: 48 89 c3                    	movq	%rax, %rbx
1000040f2: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
1000040fa: 75 23                       	jne	35 <_main+0x12af>
1000040fc: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100004104: 75 3f                       	jne	63 <_main+0x12d5>
100004106: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
10000410e: 75 5b                       	jne	91 <_main+0x12fb>
100004110: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100004118: 75 77                       	jne	119 <_main+0x1321>
10000411a: e9 91 00 00 00              	jmp	145 <_main+0x1340>
10000411f: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100004127: e8 66 2d 00 00              	callq	11622 <dyld_stub_binder+0x100006e92>
10000412c: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100004134: 74 d0                       	je	-48 <_main+0x1296>
100004136: eb 0d                       	jmp	13 <_main+0x12d5>
100004138: 48 89 c3                    	movq	%rax, %rbx
10000413b: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100004143: 74 c1                       	je	-63 <_main+0x1296>
100004145: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
10000414d: e8 40 2d 00 00              	callq	11584 <dyld_stub_binder+0x100006e92>
100004152: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
10000415a: 74 b4                       	je	-76 <_main+0x12a0>
10000415c: eb 0d                       	jmp	13 <_main+0x12fb>
10000415e: 48 89 c3                    	movq	%rax, %rbx
100004161: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100004169: 74 a5                       	je	-91 <_main+0x12a0>
10000416b: 48 8b bc 24 50 01 00 00     	movq	336(%rsp), %rdi
100004173: e8 1a 2d 00 00              	callq	11546 <dyld_stub_binder+0x100006e92>
100004178: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100004180: 75 0f                       	jne	15 <_main+0x1321>
100004182: eb 2c                       	jmp	44 <_main+0x1340>
100004184: 48 89 c3                    	movq	%rax, %rbx
100004187: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
10000418f: 74 1f                       	je	31 <_main+0x1340>
100004191: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
100004199: e8 f4 2c 00 00              	callq	11508 <dyld_stub_binder+0x100006e92>
10000419e: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000041a6: 48 85 c0                    	testq	%rax, %rax
1000041a9: 75 12                       	jne	18 <_main+0x134d>
1000041ab: eb 23                       	jmp	35 <_main+0x1360>
1000041ad: 48 89 c3                    	movq	%rax, %rbx
1000041b0: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000041b8: 48 85 c0                    	testq	%rax, %rax
1000041bb: 74 13                       	je	19 <_main+0x1360>
1000041bd: f0                          	lock
1000041be: ff 48 14                    	decl	20(%rax)
1000041c1: 75 0d                       	jne	13 <_main+0x1360>
1000041c3: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
1000041cb: e8 26 2c 00 00              	callq	11302 <dyld_stub_binder+0x100006df6>
1000041d0: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
1000041dc: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
1000041e0: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
1000041e8: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
1000041ed: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
1000041f5: 7e 21                       	jle	33 <_main+0x13a8>
1000041f7: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
1000041ff: 31 c9                       	xorl	%ecx, %ecx
100004201: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004208: 48 ff c1                    	incq	%rcx
10000420b: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
100004213: 48 39 d1                    	cmpq	%rdx, %rcx
100004216: 7c e9                       	jl	-23 <_main+0x1391>
100004218: 48 8b bc 24 c8 00 00 00     	movq	200(%rsp), %rdi
100004220: 48 8d 84 24 d0 00 00 00     	leaq	208(%rsp), %rax
100004228: 48 39 c7                    	cmpq	%rax, %rdi
10000422b: 74 21                       	je	33 <_main+0x13de>
10000422d: c5 f8 77                    	vzeroupper
100004230: e8 f7 2b 00 00              	callq	11255 <dyld_stub_binder+0x100006e2c>
100004235: eb 17                       	jmp	23 <_main+0x13de>
100004237: 48 89 c7                    	movq	%rax, %rdi
10000423a: e8 81 01 00 00              	callq	385 <_main+0x1550>
10000423f: eb 0a                       	jmp	10 <_main+0x13db>
100004241: eb 08                       	jmp	8 <_main+0x13db>
100004243: 48 89 c3                    	movq	%rax, %rbx
100004246: e9 8a 00 00 00              	jmp	138 <_main+0x1465>
10000424b: 48 89 c3                    	movq	%rax, %rbx
10000424e: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
100004256: 48 85 c0                    	testq	%rax, %rax
100004259: 74 16                       	je	22 <_main+0x1401>
10000425b: f0                          	lock
10000425c: ff 48 14                    	decl	20(%rax)
10000425f: 75 10                       	jne	16 <_main+0x1401>
100004261: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100004269: c5 f8 77                    	vzeroupper
10000426c: e8 85 2b 00 00              	callq	11141 <dyld_stub_binder+0x100006df6>
100004271: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
10000427d: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100004281: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100004289: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
10000428d: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100004295: 7e 21                       	jle	33 <_main+0x1448>
100004297: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
10000429f: 31 c9                       	xorl	%ecx, %ecx
1000042a1: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000042a8: 48 ff c1                    	incq	%rcx
1000042ab: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
1000042b3: 48 39 d1                    	cmpq	%rdx, %rcx
1000042b6: 7c e9                       	jl	-23 <_main+0x1431>
1000042b8: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
1000042c0: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
1000042c8: 48 39 c7                    	cmpq	%rax, %rdi
1000042cb: 74 08                       	je	8 <_main+0x1465>
1000042cd: c5 f8 77                    	vzeroupper
1000042d0: e8 57 2b 00 00              	callq	11095 <dyld_stub_binder+0x100006e2c>
1000042d5: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
1000042dd: c5 f8 77                    	vzeroupper
1000042e0: e8 eb 00 00 00              	callq	235 <_main+0x1560>
1000042e5: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000042ea: 48 85 c0                    	testq	%rax, %rax
1000042ed: 75 2e                       	jne	46 <_main+0x14ad>
1000042ef: eb 3f                       	jmp	63 <_main+0x14c0>
1000042f1: 48 89 c7                    	movq	%rax, %rdi
1000042f4: e8 c7 00 00 00              	callq	199 <_main+0x1550>
1000042f9: 48 89 c3                    	movq	%rax, %rbx
1000042fc: 48 8b 44 24 50              	movq	80(%rsp), %rax
100004301: 48 85 c0                    	testq	%rax, %rax
100004304: 75 17                       	jne	23 <_main+0x14ad>
100004306: eb 28                       	jmp	40 <_main+0x14c0>
100004308: 48 89 c7                    	movq	%rax, %rdi
10000430b: e8 b0 00 00 00              	callq	176 <_main+0x1550>
100004310: 48 89 c3                    	movq	%rax, %rbx
100004313: 48 8b 44 24 50              	movq	80(%rsp), %rax
100004318: 48 85 c0                    	testq	%rax, %rax
10000431b: 74 13                       	je	19 <_main+0x14c0>
10000431d: f0                          	lock
10000431e: ff 48 14                    	decl	20(%rax)
100004321: 75 0d                       	jne	13 <_main+0x14c0>
100004323: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
100004328: c5 f8 77                    	vzeroupper
10000432b: e8 c6 2a 00 00              	callq	10950 <dyld_stub_binder+0x100006df6>
100004330: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
100004339: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000433d: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
100004342: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100004347: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
10000434c: 7e 26                       	jle	38 <_main+0x1504>
10000434e: 48 8b 44 24 58              	movq	88(%rsp), %rax
100004353: 31 c9                       	xorl	%ecx, %ecx
100004355: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000435f: 90                          	nop
100004360: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004367: 48 ff c1                    	incq	%rcx
10000436a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000436f: 48 39 d1                    	cmpq	%rdx, %rcx
100004372: 7c ec                       	jl	-20 <_main+0x14f0>
100004374: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100004379: 48 8d 44 24 68              	leaq	104(%rsp), %rax
10000437e: 48 39 c7                    	cmpq	%rax, %rdi
100004381: 74 15                       	je	21 <_main+0x1528>
100004383: c5 f8 77                    	vzeroupper
100004386: e8 a1 2a 00 00              	callq	10913 <dyld_stub_binder+0x100006e2c>
10000438b: eb 0b                       	jmp	11 <_main+0x1528>
10000438d: 48 89 c7                    	movq	%rax, %rdi
100004390: e8 2b 00 00 00              	callq	43 <_main+0x1550>
100004395: 48 89 c3                    	movq	%rax, %rbx
100004398: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
1000043a0: c5 f8 77                    	vzeroupper
1000043a3: e8 42 2a 00 00              	callq	10818 <dyld_stub_binder+0x100006dea>
1000043a8: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
1000043b0: e8 bb 05 00 00              	callq	1467 <__ZN14ModelInterfaceD2Ev>
1000043b5: 48 89 df                    	movq	%rbx, %rdi
1000043b8: e8 1b 2a 00 00              	callq	10779 <dyld_stub_binder+0x100006dd8>
1000043bd: 0f 0b                       	ud2
1000043bf: 90                          	nop
1000043c0: 50                          	pushq	%rax
1000043c1: e8 de 2a 00 00              	callq	10974 <dyld_stub_binder+0x100006ea4>
1000043c6: e8 c1 2a 00 00              	callq	10945 <dyld_stub_binder+0x100006e8c>
1000043cb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000043d0: 55                          	pushq	%rbp
1000043d1: 48 89 e5                    	movq	%rsp, %rbp
1000043d4: 53                          	pushq	%rbx
1000043d5: 50                          	pushq	%rax
1000043d6: 48 89 fb                    	movq	%rdi, %rbx
1000043d9: 48 8b 87 08 01 00 00        	movq	264(%rdi), %rax
1000043e0: 48 85 c0                    	testq	%rax, %rax
1000043e3: 74 12                       	je	18 <_main+0x1587>
1000043e5: f0                          	lock
1000043e6: ff 48 14                    	decl	20(%rax)
1000043e9: 75 0c                       	jne	12 <_main+0x1587>
1000043eb: 48 8d bb d0 00 00 00        	leaq	208(%rbx), %rdi
1000043f2: e8 ff 29 00 00              	callq	10751 <dyld_stub_binder+0x100006df6>
1000043f7: 48 c7 83 08 01 00 00 00 00 00 00    	movq	$0, 264(%rbx)
100004402: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004406: c5 fc 11 83 e0 00 00 00     	vmovups	%ymm0, 224(%rbx)
10000440e: 83 bb d4 00 00 00 00        	cmpl	$0, 212(%rbx)
100004415: 7e 1f                       	jle	31 <_main+0x15c6>
100004417: 48 8b 83 10 01 00 00        	movq	272(%rbx), %rax
10000441e: 31 c9                       	xorl	%ecx, %ecx
100004420: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004427: 48 ff c1                    	incq	%rcx
10000442a: 48 63 93 d4 00 00 00        	movslq	212(%rbx), %rdx
100004431: 48 39 d1                    	cmpq	%rdx, %rcx
100004434: 7c ea                       	jl	-22 <_main+0x15b0>
100004436: 48 8b bb 18 01 00 00        	movq	280(%rbx), %rdi
10000443d: 48 8d 83 20 01 00 00        	leaq	288(%rbx), %rax
100004444: 48 39 c7                    	cmpq	%rax, %rdi
100004447: 74 08                       	je	8 <_main+0x15e1>
100004449: c5 f8 77                    	vzeroupper
10000444c: e8 db 29 00 00              	callq	10715 <dyld_stub_binder+0x100006e2c>
100004451: 48 8b 83 a8 00 00 00        	movq	168(%rbx), %rax
100004458: 48 85 c0                    	testq	%rax, %rax
10000445b: 74 12                       	je	18 <_main+0x15ff>
10000445d: f0                          	lock
10000445e: ff 48 14                    	decl	20(%rax)
100004461: 75 0c                       	jne	12 <_main+0x15ff>
100004463: 48 8d 7b 70                 	leaq	112(%rbx), %rdi
100004467: c5 f8 77                    	vzeroupper
10000446a: e8 87 29 00 00              	callq	10631 <dyld_stub_binder+0x100006df6>
10000446f: 48 c7 83 a8 00 00 00 00 00 00 00    	movq	$0, 168(%rbx)
10000447a: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000447e: c5 fc 11 83 80 00 00 00     	vmovups	%ymm0, 128(%rbx)
100004486: 83 7b 74 00                 	cmpl	$0, 116(%rbx)
10000448a: 7e 27                       	jle	39 <_main+0x1643>
10000448c: 48 8b 83 b0 00 00 00        	movq	176(%rbx), %rax
100004493: 31 c9                       	xorl	%ecx, %ecx
100004495: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000449f: 90                          	nop
1000044a0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000044a7: 48 ff c1                    	incq	%rcx
1000044aa: 48 63 53 74                 	movslq	116(%rbx), %rdx
1000044ae: 48 39 d1                    	cmpq	%rdx, %rcx
1000044b1: 7c ed                       	jl	-19 <_main+0x1630>
1000044b3: 48 8b bb b8 00 00 00        	movq	184(%rbx), %rdi
1000044ba: 48 8d 83 c0 00 00 00        	leaq	192(%rbx), %rax
1000044c1: 48 39 c7                    	cmpq	%rax, %rdi
1000044c4: 74 08                       	je	8 <_main+0x165e>
1000044c6: c5 f8 77                    	vzeroupper
1000044c9: e8 5e 29 00 00              	callq	10590 <dyld_stub_binder+0x100006e2c>
1000044ce: 48 8b 43 48                 	movq	72(%rbx), %rax
1000044d2: 48 85 c0                    	testq	%rax, %rax
1000044d5: 74 12                       	je	18 <_main+0x1679>
1000044d7: f0                          	lock
1000044d8: ff 48 14                    	decl	20(%rax)
1000044db: 75 0c                       	jne	12 <_main+0x1679>
1000044dd: 48 8d 7b 10                 	leaq	16(%rbx), %rdi
1000044e1: c5 f8 77                    	vzeroupper
1000044e4: e8 0d 29 00 00              	callq	10509 <dyld_stub_binder+0x100006df6>
1000044e9: 48 c7 43 48 00 00 00 00     	movq	$0, 72(%rbx)
1000044f1: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000044f5: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
1000044fa: 83 7b 14 00                 	cmpl	$0, 20(%rbx)
1000044fe: 7e 23                       	jle	35 <_main+0x16b3>
100004500: 48 8b 43 50                 	movq	80(%rbx), %rax
100004504: 31 c9                       	xorl	%ecx, %ecx
100004506: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004510: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004517: 48 ff c1                    	incq	%rcx
10000451a: 48 63 53 14                 	movslq	20(%rbx), %rdx
10000451e: 48 39 d1                    	cmpq	%rdx, %rcx
100004521: 7c ed                       	jl	-19 <_main+0x16a0>
100004523: 48 8b 7b 58                 	movq	88(%rbx), %rdi
100004527: 48 83 c3 60                 	addq	$96, %rbx
10000452b: 48 39 df                    	cmpq	%rbx, %rdi
10000452e: 74 08                       	je	8 <_main+0x16c8>
100004530: c5 f8 77                    	vzeroupper
100004533: e8 f4 28 00 00              	callq	10484 <dyld_stub_binder+0x100006e2c>
100004538: 48 83 c4 08                 	addq	$8, %rsp
10000453c: 5b                          	popq	%rbx
10000453d: 5d                          	popq	%rbp
10000453e: c5 f8 77                    	vzeroupper
100004541: c3                          	retq
100004542: 48 89 c7                    	movq	%rax, %rdi
100004545: e8 76 fe ff ff              	callq	-394 <_main+0x1550>
10000454a: 48 89 c7                    	movq	%rax, %rdi
10000454d: e8 6e fe ff ff              	callq	-402 <_main+0x1550>
100004552: 48 89 c7                    	movq	%rax, %rdi
100004555: e8 66 fe ff ff              	callq	-410 <_main+0x1550>
10000455a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100004560: 55                          	pushq	%rbp
100004561: 48 89 e5                    	movq	%rsp, %rbp
100004564: 41 57                       	pushq	%r15
100004566: 41 56                       	pushq	%r14
100004568: 41 55                       	pushq	%r13
10000456a: 41 54                       	pushq	%r12
10000456c: 53                          	pushq	%rbx
10000456d: 48 83 ec 28                 	subq	$40, %rsp
100004571: 49 89 d6                    	movq	%rdx, %r14
100004574: 49 89 f7                    	movq	%rsi, %r15
100004577: 48 89 fb                    	movq	%rdi, %rbx
10000457a: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
10000457e: 48 89 de                    	movq	%rbx, %rsi
100004581: e8 d6 28 00 00              	callq	10454 <dyld_stub_binder+0x100006e5c>
100004586: 80 7d b0 00                 	cmpb	$0, -80(%rbp)
10000458a: 0f 84 ae 00 00 00           	je	174 <_main+0x17ce>
100004590: 48 8b 03                    	movq	(%rbx), %rax
100004593: 48 8b 40 e8                 	movq	-24(%rax), %rax
100004597: 4c 8d 24 03                 	leaq	(%rbx,%rax), %r12
10000459b: 48 8b 7c 03 28              	movq	40(%rbx,%rax), %rdi
1000045a0: 44 8b 6c 03 08              	movl	8(%rbx,%rax), %r13d
1000045a5: 8b 84 03 90 00 00 00        	movl	144(%rbx,%rax), %eax
1000045ac: 83 f8 ff                    	cmpl	$-1, %eax
1000045af: 75 4a                       	jne	74 <_main+0x178b>
1000045b1: 48 89 7d c0                 	movq	%rdi, -64(%rbp)
1000045b5: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
1000045b9: 4c 89 e6                    	movq	%r12, %rsi
1000045bc: e8 83 28 00 00              	callq	10371 <dyld_stub_binder+0x100006e44>
1000045c1: 48 8b 35 80 4a 00 00        	movq	19072(%rip), %rsi
1000045c8: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
1000045cc: e8 6d 28 00 00              	callq	10349 <dyld_stub_binder+0x100006e3e>
1000045d1: 48 8b 08                    	movq	(%rax), %rcx
1000045d4: 48 89 c7                    	movq	%rax, %rdi
1000045d7: be 20 00 00 00              	movl	$32, %esi
1000045dc: ff 51 38                    	callq	*56(%rcx)
1000045df: 88 45 d7                    	movb	%al, -41(%rbp)
1000045e2: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
1000045e6: e8 83 28 00 00              	callq	10371 <dyld_stub_binder+0x100006e6e>
1000045eb: 0f be 45 d7                 	movsbl	-41(%rbp), %eax
1000045ef: 41 89 84 24 90 00 00 00     	movl	%eax, 144(%r12)
1000045f7: 48 8b 7d c0                 	movq	-64(%rbp), %rdi
1000045fb: 4d 01 fe                    	addq	%r15, %r14
1000045fe: 41 81 e5 b0 00 00 00        	andl	$176, %r13d
100004605: 41 83 fd 20                 	cmpl	$32, %r13d
100004609: 4c 89 fa                    	movq	%r15, %rdx
10000460c: 49 0f 44 d6                 	cmoveq	%r14, %rdx
100004610: 44 0f be c8                 	movsbl	%al, %r9d
100004614: 4c 89 fe                    	movq	%r15, %rsi
100004617: 4c 89 f1                    	movq	%r14, %rcx
10000461a: 4d 89 e0                    	movq	%r12, %r8
10000461d: e8 9e 00 00 00              	callq	158 <_main+0x1850>
100004622: 48 85 c0                    	testq	%rax, %rax
100004625: 75 17                       	jne	23 <_main+0x17ce>
100004627: 48 8b 03                    	movq	(%rbx), %rax
10000462a: 48 8b 40 e8                 	movq	-24(%rax), %rax
10000462e: 48 8d 3c 03                 	leaq	(%rbx,%rax), %rdi
100004632: 8b 74 03 20                 	movl	32(%rbx,%rax), %esi
100004636: 83 ce 05                    	orl	$5, %esi
100004639: e8 3c 28 00 00              	callq	10300 <dyld_stub_binder+0x100006e7a>
10000463e: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100004642: e8 1b 28 00 00              	callq	10267 <dyld_stub_binder+0x100006e62>
100004647: 48 89 d8                    	movq	%rbx, %rax
10000464a: 48 83 c4 28                 	addq	$40, %rsp
10000464e: 5b                          	popq	%rbx
10000464f: 41 5c                       	popq	%r12
100004651: 41 5d                       	popq	%r13
100004653: 41 5e                       	popq	%r14
100004655: 41 5f                       	popq	%r15
100004657: 5d                          	popq	%rbp
100004658: c3                          	retq
100004659: eb 0e                       	jmp	14 <_main+0x17f9>
10000465b: 49 89 c6                    	movq	%rax, %r14
10000465e: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004662: e8 07 28 00 00              	callq	10247 <dyld_stub_binder+0x100006e6e>
100004667: eb 03                       	jmp	3 <_main+0x17fc>
100004669: 49 89 c6                    	movq	%rax, %r14
10000466c: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100004670: e8 ed 27 00 00              	callq	10221 <dyld_stub_binder+0x100006e62>
100004675: eb 03                       	jmp	3 <_main+0x180a>
100004677: 49 89 c6                    	movq	%rax, %r14
10000467a: 4c 89 f7                    	movq	%r14, %rdi
10000467d: e8 22 28 00 00              	callq	10274 <dyld_stub_binder+0x100006ea4>
100004682: 48 8b 03                    	movq	(%rbx), %rax
100004685: 48 8b 78 e8                 	movq	-24(%rax), %rdi
100004689: 48 01 df                    	addq	%rbx, %rdi
10000468c: e8 e3 27 00 00              	callq	10211 <dyld_stub_binder+0x100006e74>
100004691: e8 14 28 00 00              	callq	10260 <dyld_stub_binder+0x100006eaa>
100004696: eb af                       	jmp	-81 <_main+0x17d7>
100004698: 48 89 c3                    	movq	%rax, %rbx
10000469b: e8 0a 28 00 00              	callq	10250 <dyld_stub_binder+0x100006eaa>
1000046a0: 48 89 df                    	movq	%rbx, %rdi
1000046a3: e8 30 27 00 00              	callq	10032 <dyld_stub_binder+0x100006dd8>
1000046a8: 0f 0b                       	ud2
1000046aa: 48 89 c7                    	movq	%rax, %rdi
1000046ad: e8 0e fd ff ff              	callq	-754 <_main+0x1550>
1000046b2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000046bc: 0f 1f 40 00                 	nopl	(%rax)
1000046c0: 55                          	pushq	%rbp
1000046c1: 48 89 e5                    	movq	%rsp, %rbp
1000046c4: 41 57                       	pushq	%r15
1000046c6: 41 56                       	pushq	%r14
1000046c8: 41 55                       	pushq	%r13
1000046ca: 41 54                       	pushq	%r12
1000046cc: 53                          	pushq	%rbx
1000046cd: 48 83 ec 38                 	subq	$56, %rsp
1000046d1: 48 85 ff                    	testq	%rdi, %rdi
1000046d4: 0f 84 17 01 00 00           	je	279 <_main+0x1981>
1000046da: 4d 89 c4                    	movq	%r8, %r12
1000046dd: 49 89 cf                    	movq	%rcx, %r15
1000046e0: 49 89 fe                    	movq	%rdi, %r14
1000046e3: 44 89 4d bc                 	movl	%r9d, -68(%rbp)
1000046e7: 48 89 c8                    	movq	%rcx, %rax
1000046ea: 48 29 f0                    	subq	%rsi, %rax
1000046ed: 49 8b 48 18                 	movq	24(%r8), %rcx
1000046f1: 45 31 ed                    	xorl	%r13d, %r13d
1000046f4: 48 29 c1                    	subq	%rax, %rcx
1000046f7: 4c 0f 4f e9                 	cmovgq	%rcx, %r13
1000046fb: 48 89 55 a8                 	movq	%rdx, -88(%rbp)
1000046ff: 48 89 d3                    	movq	%rdx, %rbx
100004702: 48 29 f3                    	subq	%rsi, %rbx
100004705: 48 85 db                    	testq	%rbx, %rbx
100004708: 7e 15                       	jle	21 <_main+0x18af>
10000470a: 49 8b 06                    	movq	(%r14), %rax
10000470d: 4c 89 f7                    	movq	%r14, %rdi
100004710: 48 89 da                    	movq	%rbx, %rdx
100004713: ff 50 60                    	callq	*96(%rax)
100004716: 48 39 d8                    	cmpq	%rbx, %rax
100004719: 0f 85 d2 00 00 00           	jne	210 <_main+0x1981>
10000471f: 4d 85 ed                    	testq	%r13, %r13
100004722: 0f 8e a1 00 00 00           	jle	161 <_main+0x1959>
100004728: 4c 89 65 b0                 	movq	%r12, -80(%rbp)
10000472c: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004730: c5 f8 29 45 c0              	vmovaps	%xmm0, -64(%rbp)
100004735: 48 c7 45 d0 00 00 00 00     	movq	$0, -48(%rbp)
10000473d: 49 83 fd 17                 	cmpq	$23, %r13
100004741: 73 12                       	jae	18 <_main+0x18e5>
100004743: 43 8d 44 2d 00              	leal	(%r13,%r13), %eax
100004748: 88 45 c0                    	movb	%al, -64(%rbp)
10000474b: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
10000474f: 4c 8d 65 c1                 	leaq	-63(%rbp), %r12
100004753: eb 27                       	jmp	39 <_main+0x190c>
100004755: 49 8d 5d 10                 	leaq	16(%r13), %rbx
100004759: 48 83 e3 f0                 	andq	$-16, %rbx
10000475d: 48 89 df                    	movq	%rbx, %rdi
100004760: e8 39 27 00 00              	callq	10041 <dyld_stub_binder+0x100006e9e>
100004765: 49 89 c4                    	movq	%rax, %r12
100004768: 48 89 45 d0                 	movq	%rax, -48(%rbp)
10000476c: 48 83 cb 01                 	orq	$1, %rbx
100004770: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
100004774: 4c 89 6d c8                 	movq	%r13, -56(%rbp)
100004778: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
10000477c: 0f b6 75 bc                 	movzbl	-68(%rbp), %esi
100004780: 4c 89 e7                    	movq	%r12, %rdi
100004783: 4c 89 ea                    	movq	%r13, %rdx
100004786: e8 2b 27 00 00              	callq	10027 <dyld_stub_binder+0x100006eb6>
10000478b: 43 c6 04 2c 00              	movb	$0, (%r12,%r13)
100004790: f6 45 c0 01                 	testb	$1, -64(%rbp)
100004794: 74 06                       	je	6 <_main+0x192c>
100004796: 48 8b 5d d0                 	movq	-48(%rbp), %rbx
10000479a: eb 03                       	jmp	3 <_main+0x192f>
10000479c: 48 ff c3                    	incq	%rbx
10000479f: 4c 8b 65 b0                 	movq	-80(%rbp), %r12
1000047a3: 49 8b 06                    	movq	(%r14), %rax
1000047a6: 4c 89 f7                    	movq	%r14, %rdi
1000047a9: 48 89 de                    	movq	%rbx, %rsi
1000047ac: 4c 89 ea                    	movq	%r13, %rdx
1000047af: ff 50 60                    	callq	*96(%rax)
1000047b2: 48 89 c3                    	movq	%rax, %rbx
1000047b5: f6 45 c0 01                 	testb	$1, -64(%rbp)
1000047b9: 74 09                       	je	9 <_main+0x1954>
1000047bb: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
1000047bf: e8 ce 26 00 00              	callq	9934 <dyld_stub_binder+0x100006e92>
1000047c4: 4c 39 eb                    	cmpq	%r13, %rbx
1000047c7: 75 28                       	jne	40 <_main+0x1981>
1000047c9: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
1000047cd: 49 29 f7                    	subq	%rsi, %r15
1000047d0: 4d 85 ff                    	testq	%r15, %r15
1000047d3: 7e 11                       	jle	17 <_main+0x1976>
1000047d5: 49 8b 06                    	movq	(%r14), %rax
1000047d8: 4c 89 f7                    	movq	%r14, %rdi
1000047db: 4c 89 fa                    	movq	%r15, %rdx
1000047de: ff 50 60                    	callq	*96(%rax)
1000047e1: 4c 39 f8                    	cmpq	%r15, %rax
1000047e4: 75 0b                       	jne	11 <_main+0x1981>
1000047e6: 49 c7 44 24 18 00 00 00 00  	movq	$0, 24(%r12)
1000047ef: eb 03                       	jmp	3 <_main+0x1984>
1000047f1: 45 31 f6                    	xorl	%r14d, %r14d
1000047f4: 4c 89 f0                    	movq	%r14, %rax
1000047f7: 48 83 c4 38                 	addq	$56, %rsp
1000047fb: 5b                          	popq	%rbx
1000047fc: 41 5c                       	popq	%r12
1000047fe: 41 5d                       	popq	%r13
100004800: 41 5e                       	popq	%r14
100004802: 41 5f                       	popq	%r15
100004804: 5d                          	popq	%rbp
100004805: c3                          	retq
100004806: 48 89 c3                    	movq	%rax, %rbx
100004809: f6 45 c0 01                 	testb	$1, -64(%rbp)
10000480d: 74 09                       	je	9 <_main+0x19a8>
10000480f: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
100004813: e8 7a 26 00 00              	callq	9850 <dyld_stub_binder+0x100006e92>
100004818: 48 89 df                    	movq	%rbx, %rdi
10000481b: e8 b8 25 00 00              	callq	9656 <dyld_stub_binder+0x100006dd8>
100004820: 0f 0b                       	ud2
100004822: 90                          	nop
100004823: 90                          	nop
100004824: 90                          	nop
100004825: 90                          	nop
100004826: 90                          	nop
100004827: 90                          	nop
100004828: 90                          	nop
100004829: 90                          	nop
10000482a: 90                          	nop
10000482b: 90                          	nop
10000482c: 90                          	nop
10000482d: 90                          	nop
10000482e: 90                          	nop
10000482f: 90                          	nop
100004830: 55                          	pushq	%rbp
100004831: 48 89 e5                    	movq	%rsp, %rbp
100004834: 48 8b 05 c5 47 00 00        	movq	18373(%rip), %rax
10000483b: 80 38 00                    	cmpb	$0, (%rax)
10000483e: 74 02                       	je	2 <_main+0x19d2>
100004840: 5d                          	popq	%rbp
100004841: c3                          	retq
100004842: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004849: 5d                          	popq	%rbp
10000484a: c3                          	retq
10000484b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004850: 55                          	pushq	%rbp
100004851: 48 89 e5                    	movq	%rsp, %rbp
100004854: 48 8b 05 c5 47 00 00        	movq	18373(%rip), %rax
10000485b: 80 38 00                    	cmpb	$0, (%rax)
10000485e: 74 02                       	je	2 <_main+0x19f2>
100004860: 5d                          	popq	%rbp
100004861: c3                          	retq
100004862: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004869: 5d                          	popq	%rbp
10000486a: c3                          	retq
10000486b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004870: 55                          	pushq	%rbp
100004871: 48 89 e5                    	movq	%rsp, %rbp
100004874: 48 8b 05 bd 47 00 00        	movq	18365(%rip), %rax
10000487b: 80 38 00                    	cmpb	$0, (%rax)
10000487e: 74 02                       	je	2 <_main+0x1a12>
100004880: 5d                          	popq	%rbp
100004881: c3                          	retq
100004882: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004889: 5d                          	popq	%rbp
10000488a: c3                          	retq
10000488b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004890: 55                          	pushq	%rbp
100004891: 48 89 e5                    	movq	%rsp, %rbp
100004894: 48 8b 05 95 47 00 00        	movq	18325(%rip), %rax
10000489b: 80 38 00                    	cmpb	$0, (%rax)
10000489e: 74 02                       	je	2 <_main+0x1a32>
1000048a0: 5d                          	popq	%rbp
1000048a1: c3                          	retq
1000048a2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048a9: 5d                          	popq	%rbp
1000048aa: c3                          	retq
1000048ab: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048b0: 55                          	pushq	%rbp
1000048b1: 48 89 e5                    	movq	%rsp, %rbp
1000048b4: 48 8b 05 6d 47 00 00        	movq	18285(%rip), %rax
1000048bb: 80 38 00                    	cmpb	$0, (%rax)
1000048be: 74 02                       	je	2 <_main+0x1a52>
1000048c0: 5d                          	popq	%rbp
1000048c1: c3                          	retq
1000048c2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048c9: 5d                          	popq	%rbp
1000048ca: c3                          	retq
1000048cb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048d0: 55                          	pushq	%rbp
1000048d1: 48 89 e5                    	movq	%rsp, %rbp
1000048d4: 48 8b 05 2d 47 00 00        	movq	18221(%rip), %rax
1000048db: 80 38 00                    	cmpb	$0, (%rax)
1000048de: 74 02                       	je	2 <_main+0x1a72>
1000048e0: 5d                          	popq	%rbp
1000048e1: c3                          	retq
1000048e2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048e9: 5d                          	popq	%rbp
1000048ea: c3                          	retq
1000048eb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048f0: 55                          	pushq	%rbp
1000048f1: 48 89 e5                    	movq	%rsp, %rbp
1000048f4: 48 8b 05 15 47 00 00        	movq	18197(%rip), %rax
1000048fb: 80 38 00                    	cmpb	$0, (%rax)
1000048fe: 74 02                       	je	2 <_main+0x1a92>
100004900: 5d                          	popq	%rbp
100004901: c3                          	retq
100004902: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004909: 5d                          	popq	%rbp
10000490a: c3                          	retq
10000490b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004910: 55                          	pushq	%rbp
100004911: 48 89 e5                    	movq	%rsp, %rbp
100004914: 48 8b 05 fd 46 00 00        	movq	18173(%rip), %rax
10000491b: 80 38 00                    	cmpb	$0, (%rax)
10000491e: 74 02                       	je	2 <_main+0x1ab2>
100004920: 5d                          	popq	%rbp
100004921: c3                          	retq
100004922: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004929: 5d                          	popq	%rbp
10000492a: c3                          	retq
10000492b: 90                          	nop
10000492c: 90                          	nop
10000492d: 90                          	nop
10000492e: 90                          	nop
10000492f: 90                          	nop

0000000100004930 __ZN14ModelInterfaceC2Ev:
100004930: 55                          	pushq	%rbp
100004931: 48 89 e5                    	movq	%rsp, %rbp
100004934: 48 8d 05 7d 47 00 00        	leaq	18301(%rip), %rax
10000493b: 48 89 07                    	movq	%rax, (%rdi)
10000493e: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004942: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
100004947: 5d                          	popq	%rbp
100004948: c3                          	retq
100004949: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004950 __ZN14ModelInterfaceC1Ev:
100004950: 55                          	pushq	%rbp
100004951: 48 89 e5                    	movq	%rsp, %rbp
100004954: 48 8d 05 5d 47 00 00        	leaq	18269(%rip), %rax
10000495b: 48 89 07                    	movq	%rax, (%rdi)
10000495e: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004962: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
100004967: 5d                          	popq	%rbp
100004968: c3                          	retq
100004969: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004970 __ZN14ModelInterfaceD2Ev:
100004970: 55                          	pushq	%rbp
100004971: 48 89 e5                    	movq	%rsp, %rbp
100004974: 53                          	pushq	%rbx
100004975: 50                          	pushq	%rax
100004976: 48 89 fb                    	movq	%rdi, %rbx
100004979: 48 8d 05 38 47 00 00        	leaq	18232(%rip), %rax
100004980: 48 89 07                    	movq	%rax, (%rdi)
100004983: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100004987: 48 85 ff                    	testq	%rdi, %rdi
10000498a: 74 05                       	je	5 <__ZN14ModelInterfaceD2Ev+0x21>
10000498c: e8 01 25 00 00              	callq	9473 <dyld_stub_binder+0x100006e92>
100004991: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100004995: 48 83 c4 08                 	addq	$8, %rsp
100004999: 48 85 ff                    	testq	%rdi, %rdi
10000499c: 74 07                       	je	7 <__ZN14ModelInterfaceD2Ev+0x35>
10000499e: 5b                          	popq	%rbx
10000499f: 5d                          	popq	%rbp
1000049a0: e9 ed 24 00 00              	jmp	9453 <dyld_stub_binder+0x100006e92>
1000049a5: 5b                          	popq	%rbx
1000049a6: 5d                          	popq	%rbp
1000049a7: c3                          	retq
1000049a8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

00000001000049b0 __ZN14ModelInterfaceD1Ev:
1000049b0: 55                          	pushq	%rbp
1000049b1: 48 89 e5                    	movq	%rsp, %rbp
1000049b4: 53                          	pushq	%rbx
1000049b5: 50                          	pushq	%rax
1000049b6: 48 89 fb                    	movq	%rdi, %rbx
1000049b9: 48 8d 05 f8 46 00 00        	leaq	18168(%rip), %rax
1000049c0: 48 89 07                    	movq	%rax, (%rdi)
1000049c3: 48 8b 7f 28                 	movq	40(%rdi), %rdi
1000049c7: 48 85 ff                    	testq	%rdi, %rdi
1000049ca: 74 05                       	je	5 <__ZN14ModelInterfaceD1Ev+0x21>
1000049cc: e8 c1 24 00 00              	callq	9409 <dyld_stub_binder+0x100006e92>
1000049d1: 48 8b 7b 30                 	movq	48(%rbx), %rdi
1000049d5: 48 83 c4 08                 	addq	$8, %rsp
1000049d9: 48 85 ff                    	testq	%rdi, %rdi
1000049dc: 74 07                       	je	7 <__ZN14ModelInterfaceD1Ev+0x35>
1000049de: 5b                          	popq	%rbx
1000049df: 5d                          	popq	%rbp
1000049e0: e9 ad 24 00 00              	jmp	9389 <dyld_stub_binder+0x100006e92>
1000049e5: 5b                          	popq	%rbx
1000049e6: 5d                          	popq	%rbp
1000049e7: c3                          	retq
1000049e8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

00000001000049f0 __ZN14ModelInterfaceD0Ev:
1000049f0: 55                          	pushq	%rbp
1000049f1: 48 89 e5                    	movq	%rsp, %rbp
1000049f4: 53                          	pushq	%rbx
1000049f5: 50                          	pushq	%rax
1000049f6: 48 89 fb                    	movq	%rdi, %rbx
1000049f9: 48 8d 05 b8 46 00 00        	leaq	18104(%rip), %rax
100004a00: 48 89 07                    	movq	%rax, (%rdi)
100004a03: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100004a07: 48 85 ff                    	testq	%rdi, %rdi
100004a0a: 74 05                       	je	5 <__ZN14ModelInterfaceD0Ev+0x21>
100004a0c: e8 81 24 00 00              	callq	9345 <dyld_stub_binder+0x100006e92>
100004a11: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100004a15: 48 85 ff                    	testq	%rdi, %rdi
100004a18: 74 05                       	je	5 <__ZN14ModelInterfaceD0Ev+0x2f>
100004a1a: e8 73 24 00 00              	callq	9331 <dyld_stub_binder+0x100006e92>
100004a1f: 48 89 df                    	movq	%rbx, %rdi
100004a22: 48 83 c4 08                 	addq	$8, %rsp
100004a26: 5b                          	popq	%rbx
100004a27: 5d                          	popq	%rbp
100004a28: e9 65 24 00 00              	jmp	9317 <dyld_stub_binder+0x100006e92>
100004a2d: 0f 1f 00                    	nopl	(%rax)

0000000100004a30 __ZN14ModelInterface7forwardEv:
100004a30: 55                          	pushq	%rbp
100004a31: 48 89 e5                    	movq	%rsp, %rbp
100004a34: 5d                          	popq	%rbp
100004a35: c3                          	retq
100004a36: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

0000000100004a40 __ZN14ModelInterface12input_bufferEv:
100004a40: 55                          	pushq	%rbp
100004a41: 48 89 e5                    	movq	%rsp, %rbp
100004a44: 0f b6 47 24                 	movzbl	36(%rdi), %eax
100004a48: 48 8b 44 c7 28              	movq	40(%rdi,%rax,8), %rax
100004a4d: 5d                          	popq	%rbp
100004a4e: c3                          	retq
100004a4f: 90                          	nop

0000000100004a50 __ZN14ModelInterface13output_bufferEv:
100004a50: 55                          	pushq	%rbp
100004a51: 48 89 e5                    	movq	%rsp, %rbp
100004a54: 31 c0                       	xorl	%eax, %eax
100004a56: 80 7f 24 00                 	cmpb	$0, 36(%rdi)
100004a5a: 0f 94 c0                    	sete	%al
100004a5d: 48 8b 44 c7 28              	movq	40(%rdi,%rax,8), %rax
100004a62: 5d                          	popq	%rbp
100004a63: c3                          	retq
100004a64: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004a6e: 66 90                       	nop

0000000100004a70 __ZN14ModelInterface11init_bufferEj:
100004a70: 55                          	pushq	%rbp
100004a71: 48 89 e5                    	movq	%rsp, %rbp
100004a74: 41 57                       	pushq	%r15
100004a76: 41 56                       	pushq	%r14
100004a78: 41 54                       	pushq	%r12
100004a7a: 53                          	pushq	%rbx
100004a7b: 41 89 f7                    	movl	%esi, %r15d
100004a7e: 48 89 fb                    	movq	%rdi, %rbx
100004a81: c6 47 24 00                 	movb	$0, 36(%rdi)
100004a85: 41 89 f6                    	movl	%esi, %r14d
100004a88: 4c 89 f7                    	movq	%r14, %rdi
100004a8b: e8 08 24 00 00              	callq	9224 <dyld_stub_binder+0x100006e98>
100004a90: 49 89 c4                    	movq	%rax, %r12
100004a93: 48 89 43 28                 	movq	%rax, 40(%rbx)
100004a97: 4c 89 f7                    	movq	%r14, %rdi
100004a9a: e8 f9 23 00 00              	callq	9209 <dyld_stub_binder+0x100006e98>
100004a9f: 48 89 43 30                 	movq	%rax, 48(%rbx)
100004aa3: 45 85 ff                    	testl	%r15d, %r15d
100004aa6: 0f 84 44 01 00 00           	je	324 <__ZN14ModelInterface11init_bufferEj+0x180>
100004aac: 41 c6 04 24 00              	movb	$0, (%r12)
100004ab1: 41 83 ff 01                 	cmpl	$1, %r15d
100004ab5: 0f 84 95 00 00 00           	je	149 <__ZN14ModelInterface11init_bufferEj+0xe0>
100004abb: 41 8d 46 ff                 	leal	-1(%r14), %eax
100004abf: 49 8d 56 fe                 	leaq	-2(%r14), %rdx
100004ac3: 83 e0 07                    	andl	$7, %eax
100004ac6: b9 01 00 00 00              	movl	$1, %ecx
100004acb: 48 83 fa 07                 	cmpq	$7, %rdx
100004acf: 72 63                       	jb	99 <__ZN14ModelInterface11init_bufferEj+0xc4>
100004ad1: 48 89 c2                    	movq	%rax, %rdx
100004ad4: 48 f7 d2                    	notq	%rdx
100004ad7: 4c 01 f2                    	addq	%r14, %rdx
100004ada: 31 c9                       	xorl	%ecx, %ecx
100004adc: 0f 1f 40 00                 	nopl	(%rax)
100004ae0: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004ae4: c6 44 0e 01 00              	movb	$0, 1(%rsi,%rcx)
100004ae9: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004aed: c6 44 0e 02 00              	movb	$0, 2(%rsi,%rcx)
100004af2: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004af6: c6 44 0e 03 00              	movb	$0, 3(%rsi,%rcx)
100004afb: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004aff: c6 44 0e 04 00              	movb	$0, 4(%rsi,%rcx)
100004b04: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b08: c6 44 0e 05 00              	movb	$0, 5(%rsi,%rcx)
100004b0d: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b11: c6 44 0e 06 00              	movb	$0, 6(%rsi,%rcx)
100004b16: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b1a: c6 44 0e 07 00              	movb	$0, 7(%rsi,%rcx)
100004b1f: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b23: c6 44 0e 08 00              	movb	$0, 8(%rsi,%rcx)
100004b28: 48 83 c1 08                 	addq	$8, %rcx
100004b2c: 48 39 ca                    	cmpq	%rcx, %rdx
100004b2f: 75 af                       	jne	-81 <__ZN14ModelInterface11init_bufferEj+0x70>
100004b31: 48 ff c1                    	incq	%rcx
100004b34: 48 85 c0                    	testq	%rax, %rax
100004b37: 74 17                       	je	23 <__ZN14ModelInterface11init_bufferEj+0xe0>
100004b39: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004b40: 48 8b 53 28                 	movq	40(%rbx), %rdx
100004b44: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004b48: 48 ff c1                    	incq	%rcx
100004b4b: 48 ff c8                    	decq	%rax
100004b4e: 75 f0                       	jne	-16 <__ZN14ModelInterface11init_bufferEj+0xd0>
100004b50: 45 85 ff                    	testl	%r15d, %r15d
100004b53: 0f 84 97 00 00 00           	je	151 <__ZN14ModelInterface11init_bufferEj+0x180>
100004b59: 49 8d 4e ff                 	leaq	-1(%r14), %rcx
100004b5d: 44 89 f0                    	movl	%r14d, %eax
100004b60: 83 e0 07                    	andl	$7, %eax
100004b63: 48 83 f9 07                 	cmpq	$7, %rcx
100004b67: 73 0c                       	jae	12 <__ZN14ModelInterface11init_bufferEj+0x105>
100004b69: 31 c9                       	xorl	%ecx, %ecx
100004b6b: 48 85 c0                    	testq	%rax, %rax
100004b6e: 75 70                       	jne	112 <__ZN14ModelInterface11init_bufferEj+0x170>
100004b70: e9 7b 00 00 00              	jmp	123 <__ZN14ModelInterface11init_bufferEj+0x180>
100004b75: 49 29 c6                    	subq	%rax, %r14
100004b78: 31 c9                       	xorl	%ecx, %ecx
100004b7a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100004b80: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b84: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004b88: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b8c: c6 44 0a 01 00              	movb	$0, 1(%rdx,%rcx)
100004b91: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b95: c6 44 0a 02 00              	movb	$0, 2(%rdx,%rcx)
100004b9a: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004b9e: c6 44 0a 03 00              	movb	$0, 3(%rdx,%rcx)
100004ba3: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004ba7: c6 44 0a 04 00              	movb	$0, 4(%rdx,%rcx)
100004bac: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004bb0: c6 44 0a 05 00              	movb	$0, 5(%rdx,%rcx)
100004bb5: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004bb9: c6 44 0a 06 00              	movb	$0, 6(%rdx,%rcx)
100004bbe: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004bc2: c6 44 0a 07 00              	movb	$0, 7(%rdx,%rcx)
100004bc7: 48 83 c1 08                 	addq	$8, %rcx
100004bcb: 49 39 ce                    	cmpq	%rcx, %r14
100004bce: 75 b0                       	jne	-80 <__ZN14ModelInterface11init_bufferEj+0x110>
100004bd0: 48 85 c0                    	testq	%rax, %rax
100004bd3: 74 1b                       	je	27 <__ZN14ModelInterface11init_bufferEj+0x180>
100004bd5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004bdf: 90                          	nop
100004be0: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004be4: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004be8: 48 ff c1                    	incq	%rcx
100004beb: 48 ff c8                    	decq	%rax
100004bee: 75 f0                       	jne	-16 <__ZN14ModelInterface11init_bufferEj+0x170>
100004bf0: 5b                          	popq	%rbx
100004bf1: 41 5c                       	popq	%r12
100004bf3: 41 5e                       	popq	%r14
100004bf5: 41 5f                       	popq	%r15
100004bf7: 5d                          	popq	%rbp
100004bf8: c3                          	retq
100004bf9: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004c00 __ZN14ModelInterface11swap_bufferEv:
100004c00: 55                          	pushq	%rbp
100004c01: 48 89 e5                    	movq	%rsp, %rbp
100004c04: 80 77 24 01                 	xorb	$1, 36(%rdi)
100004c08: 5d                          	popq	%rbp
100004c09: c3                          	retq
100004c0a: 90                          	nop
100004c0b: 90                          	nop
100004c0c: 90                          	nop
100004c0d: 90                          	nop
100004c0e: 90                          	nop
100004c0f: 90                          	nop

0000000100004c10 __Z4ReLUPaS_j:
100004c10: 55                          	pushq	%rbp
100004c11: 48 89 e5                    	movq	%rsp, %rbp
100004c14: 83 fa 04                    	cmpl	$4, %edx
100004c17: 0f 82 88 00 00 00           	jb	136 <__Z4ReLUPaS_j+0x95>
100004c1d: 8d 42 fc                    	leal	-4(%rdx), %eax
100004c20: 41 89 c2                    	movl	%eax, %r10d
100004c23: 41 c1 ea 02                 	shrl	$2, %r10d
100004c27: 41 ff c2                    	incl	%r10d
100004c2a: 41 83 fa 1f                 	cmpl	$31, %r10d
100004c2e: 76 24                       	jbe	36 <__Z4ReLUPaS_j+0x44>
100004c30: 83 e0 fc                    	andl	$-4, %eax
100004c33: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100004c37: 48 83 c1 04                 	addq	$4, %rcx
100004c3b: 48 39 f9                    	cmpq	%rdi, %rcx
100004c3e: 0f 86 78 02 00 00           	jbe	632 <__Z4ReLUPaS_j+0x2ac>
100004c44: 48 01 f8                    	addq	%rdi, %rax
100004c47: 48 83 c0 04                 	addq	$4, %rax
100004c4b: 48 39 f0                    	cmpq	%rsi, %rax
100004c4e: 0f 86 68 02 00 00           	jbe	616 <__Z4ReLUPaS_j+0x2ac>
100004c54: 89 d0                       	movl	%edx, %eax
100004c56: 45 31 c0                    	xorl	%r8d, %r8d
100004c59: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004c60: 0f b6 0e                    	movzbl	(%rsi), %ecx
100004c63: 84 c9                       	testb	%cl, %cl
100004c65: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c69: 88 0f                       	movb	%cl, (%rdi)
100004c6b: 0f b6 4e 01                 	movzbl	1(%rsi), %ecx
100004c6f: 84 c9                       	testb	%cl, %cl
100004c71: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c75: 88 4f 01                    	movb	%cl, 1(%rdi)
100004c78: 0f b6 4e 02                 	movzbl	2(%rsi), %ecx
100004c7c: 84 c9                       	testb	%cl, %cl
100004c7e: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c82: 88 4f 02                    	movb	%cl, 2(%rdi)
100004c85: 0f b6 4e 03                 	movzbl	3(%rsi), %ecx
100004c89: 84 c9                       	testb	%cl, %cl
100004c8b: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004c8f: 88 4f 03                    	movb	%cl, 3(%rdi)
100004c92: 48 83 c7 04                 	addq	$4, %rdi
100004c96: 48 83 c6 04                 	addq	$4, %rsi
100004c9a: 83 c0 fc                    	addl	$-4, %eax
100004c9d: 83 f8 03                    	cmpl	$3, %eax
100004ca0: 77 be                       	ja	-66 <__Z4ReLUPaS_j+0x50>
100004ca2: 83 e2 03                    	andl	$3, %edx
100004ca5: 85 d2                       	testl	%edx, %edx
100004ca7: 0f 84 0a 02 00 00           	je	522 <__Z4ReLUPaS_j+0x2a7>
100004cad: 8d 42 ff                    	leal	-1(%rdx), %eax
100004cb0: 4c 8d 50 01                 	leaq	1(%rax), %r10
100004cb4: 49 83 fa 7f                 	cmpq	$127, %r10
100004cb8: 0f 86 2e 01 00 00           	jbe	302 <__Z4ReLUPaS_j+0x1dc>
100004cbe: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100004cc2: 48 83 c1 01                 	addq	$1, %rcx
100004cc6: 48 39 cf                    	cmpq	%rcx, %rdi
100004cc9: 73 10                       	jae	16 <__Z4ReLUPaS_j+0xcb>
100004ccb: 48 01 f8                    	addq	%rdi, %rax
100004cce: 48 83 c0 01                 	addq	$1, %rax
100004cd2: 48 39 c6                    	cmpq	%rax, %rsi
100004cd5: 0f 82 11 01 00 00           	jb	273 <__Z4ReLUPaS_j+0x1dc>
100004cdb: 4d 89 d0                    	movq	%r10, %r8
100004cde: 49 83 e0 80                 	andq	$-128, %r8
100004ce2: 49 8d 40 80                 	leaq	-128(%r8), %rax
100004ce6: 48 89 c1                    	movq	%rax, %rcx
100004ce9: 48 c1 e9 07                 	shrq	$7, %rcx
100004ced: 48 ff c1                    	incq	%rcx
100004cf0: 41 89 c9                    	movl	%ecx, %r9d
100004cf3: 41 83 e1 01                 	andl	$1, %r9d
100004cf7: 48 85 c0                    	testq	%rax, %rax
100004cfa: 0f 84 0f 09 00 00           	je	2319 <__Z4ReLUPaS_j+0x9ff>
100004d00: 4c 89 c8                    	movq	%r9, %rax
100004d03: 48 29 c8                    	subq	%rcx, %rax
100004d06: 31 c9                       	xorl	%ecx, %ecx
100004d08: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100004d0c: 0f 1f 40 00                 	nopl	(%rax)
100004d10: c4 e2 7d 3c 0c 0e           	vpmaxsb	(%rsi,%rcx), %ymm0, %ymm1
100004d16: c4 e2 7d 3c 54 0e 20        	vpmaxsb	32(%rsi,%rcx), %ymm0, %ymm2
100004d1d: c4 e2 7d 3c 5c 0e 40        	vpmaxsb	64(%rsi,%rcx), %ymm0, %ymm3
100004d24: c4 e2 7d 3c 64 0e 60        	vpmaxsb	96(%rsi,%rcx), %ymm0, %ymm4
100004d2b: c5 fe 7f 0c 0f              	vmovdqu	%ymm1, (%rdi,%rcx)
100004d30: c5 fe 7f 54 0f 20           	vmovdqu	%ymm2, 32(%rdi,%rcx)
100004d36: c5 fe 7f 5c 0f 40           	vmovdqu	%ymm3, 64(%rdi,%rcx)
100004d3c: c5 fe 7f 64 0f 60           	vmovdqu	%ymm4, 96(%rdi,%rcx)
100004d42: c4 e2 7d 3c 8c 0e 80 00 00 00       	vpmaxsb	128(%rsi,%rcx), %ymm0, %ymm1
100004d4c: c4 e2 7d 3c 94 0e a0 00 00 00       	vpmaxsb	160(%rsi,%rcx), %ymm0, %ymm2
100004d56: c4 e2 7d 3c 9c 0e c0 00 00 00       	vpmaxsb	192(%rsi,%rcx), %ymm0, %ymm3
100004d60: c4 e2 7d 3c a4 0e e0 00 00 00       	vpmaxsb	224(%rsi,%rcx), %ymm0, %ymm4
100004d6a: c5 fe 7f 8c 0f 80 00 00 00  	vmovdqu	%ymm1, 128(%rdi,%rcx)
100004d73: c5 fe 7f 94 0f a0 00 00 00  	vmovdqu	%ymm2, 160(%rdi,%rcx)
100004d7c: c5 fe 7f 9c 0f c0 00 00 00  	vmovdqu	%ymm3, 192(%rdi,%rcx)
100004d85: c5 fe 7f a4 0f e0 00 00 00  	vmovdqu	%ymm4, 224(%rdi,%rcx)
100004d8e: 48 81 c1 00 01 00 00        	addq	$256, %rcx
100004d95: 48 83 c0 02                 	addq	$2, %rax
100004d99: 0f 85 71 ff ff ff           	jne	-143 <__Z4ReLUPaS_j+0x100>
100004d9f: 4d 85 c9                    	testq	%r9, %r9
100004da2: 74 36                       	je	54 <__Z4ReLUPaS_j+0x1ca>
100004da4: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100004da8: c4 e2 7d 3c 0c 0e           	vpmaxsb	(%rsi,%rcx), %ymm0, %ymm1
100004dae: c4 e2 7d 3c 54 0e 20        	vpmaxsb	32(%rsi,%rcx), %ymm0, %ymm2
100004db5: c4 e2 7d 3c 5c 0e 40        	vpmaxsb	64(%rsi,%rcx), %ymm0, %ymm3
100004dbc: c4 e2 7d 3c 44 0e 60        	vpmaxsb	96(%rsi,%rcx), %ymm0, %ymm0
100004dc3: c5 fe 7f 0c 0f              	vmovdqu	%ymm1, (%rdi,%rcx)
100004dc8: c5 fe 7f 54 0f 20           	vmovdqu	%ymm2, 32(%rdi,%rcx)
100004dce: c5 fe 7f 5c 0f 40           	vmovdqu	%ymm3, 64(%rdi,%rcx)
100004dd4: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
100004dda: 4d 39 c2                    	cmpq	%r8, %r10
100004ddd: 0f 84 d4 00 00 00           	je	212 <__Z4ReLUPaS_j+0x2a7>
100004de3: 44 29 c2                    	subl	%r8d, %edx
100004de6: 4c 01 c6                    	addq	%r8, %rsi
100004de9: 4c 01 c7                    	addq	%r8, %rdi
100004dec: 44 8d 42 ff                 	leal	-1(%rdx), %r8d
100004df0: f6 c2 07                    	testb	$7, %dl
100004df3: 74 38                       	je	56 <__Z4ReLUPaS_j+0x21d>
100004df5: 41 89 d2                    	movl	%edx, %r10d
100004df8: 41 83 e2 07                 	andl	$7, %r10d
100004dfc: 45 31 c9                    	xorl	%r9d, %r9d
100004dff: 31 c9                       	xorl	%ecx, %ecx
100004e01: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004e0b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004e10: 0f b6 04 0e                 	movzbl	(%rsi,%rcx), %eax
100004e14: 84 c0                       	testb	%al, %al
100004e16: 41 0f 48 c1                 	cmovsl	%r9d, %eax
100004e1a: 88 04 0f                    	movb	%al, (%rdi,%rcx)
100004e1d: 48 ff c1                    	incq	%rcx
100004e20: 41 39 ca                    	cmpl	%ecx, %r10d
100004e23: 75 eb                       	jne	-21 <__Z4ReLUPaS_j+0x200>
100004e25: 29 ca                       	subl	%ecx, %edx
100004e27: 48 01 ce                    	addq	%rcx, %rsi
100004e2a: 48 01 cf                    	addq	%rcx, %rdi
100004e2d: 41 83 f8 07                 	cmpl	$7, %r8d
100004e31: 0f 82 80 00 00 00           	jb	128 <__Z4ReLUPaS_j+0x2a7>
100004e37: 41 89 d0                    	movl	%edx, %r8d
100004e3a: 31 c9                       	xorl	%ecx, %ecx
100004e3c: 31 d2                       	xorl	%edx, %edx
100004e3e: 66 90                       	nop
100004e40: 0f b6 04 16                 	movzbl	(%rsi,%rdx), %eax
100004e44: 84 c0                       	testb	%al, %al
100004e46: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e49: 88 04 17                    	movb	%al, (%rdi,%rdx)
100004e4c: 0f b6 44 16 01              	movzbl	1(%rsi,%rdx), %eax
100004e51: 84 c0                       	testb	%al, %al
100004e53: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e56: 88 44 17 01                 	movb	%al, 1(%rdi,%rdx)
100004e5a: 0f b6 44 16 02              	movzbl	2(%rsi,%rdx), %eax
100004e5f: 84 c0                       	testb	%al, %al
100004e61: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e64: 88 44 17 02                 	movb	%al, 2(%rdi,%rdx)
100004e68: 0f b6 44 16 03              	movzbl	3(%rsi,%rdx), %eax
100004e6d: 84 c0                       	testb	%al, %al
100004e6f: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e72: 88 44 17 03                 	movb	%al, 3(%rdi,%rdx)
100004e76: 0f b6 44 16 04              	movzbl	4(%rsi,%rdx), %eax
100004e7b: 84 c0                       	testb	%al, %al
100004e7d: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e80: 88 44 17 04                 	movb	%al, 4(%rdi,%rdx)
100004e84: 0f b6 44 16 05              	movzbl	5(%rsi,%rdx), %eax
100004e89: 84 c0                       	testb	%al, %al
100004e8b: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e8e: 88 44 17 05                 	movb	%al, 5(%rdi,%rdx)
100004e92: 0f b6 44 16 06              	movzbl	6(%rsi,%rdx), %eax
100004e97: 84 c0                       	testb	%al, %al
100004e99: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e9c: 88 44 17 06                 	movb	%al, 6(%rdi,%rdx)
100004ea0: 0f b6 44 16 07              	movzbl	7(%rsi,%rdx), %eax
100004ea5: 84 c0                       	testb	%al, %al
100004ea7: 0f 48 c1                    	cmovsl	%ecx, %eax
100004eaa: 88 44 17 07                 	movb	%al, 7(%rdi,%rdx)
100004eae: 48 83 c2 08                 	addq	$8, %rdx
100004eb2: 41 39 d0                    	cmpl	%edx, %r8d
100004eb5: 75 89                       	jne	-119 <__Z4ReLUPaS_j+0x230>
100004eb7: 5d                          	popq	%rbp
100004eb8: c5 f8 77                    	vzeroupper
100004ebb: c3                          	retq
100004ebc: 45 89 d0                    	movl	%r10d, %r8d
100004ebf: 41 83 e0 e0                 	andl	$-32, %r8d
100004ec3: 49 8d 40 e0                 	leaq	-32(%r8), %rax
100004ec7: 48 89 c1                    	movq	%rax, %rcx
100004eca: 48 c1 e9 05                 	shrq	$5, %rcx
100004ece: 48 ff c1                    	incq	%rcx
100004ed1: 41 89 c9                    	movl	%ecx, %r9d
100004ed4: 41 83 e1 01                 	andl	$1, %r9d
100004ed8: 48 85 c0                    	testq	%rax, %rax
100004edb: 0f 84 3e 07 00 00           	je	1854 <__Z4ReLUPaS_j+0xa0f>
100004ee1: 4c 89 c8                    	movq	%r9, %rax
100004ee4: 48 29 c8                    	subq	%rcx, %rax
100004ee7: 31 c9                       	xorl	%ecx, %ecx
100004ee9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004ef0: c5 7a 6f 34 0e              	vmovdqu	(%rsi,%rcx), %xmm14
100004ef5: c5 7a 6f 7c 0e 10           	vmovdqu	16(%rsi,%rcx), %xmm15
100004efb: c5 fa 6f 54 0e 20           	vmovdqu	32(%rsi,%rcx), %xmm2
100004f01: c5 fa 6f 5c 0e 30           	vmovdqu	48(%rsi,%rcx), %xmm3
100004f07: c5 79 6f 1d 11 22 00 00     	vmovdqa	8721(%rip), %xmm11
100004f0f: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100004f14: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
100004f19: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100004f1d: c5 79 6f 05 0b 22 00 00     	vmovdqa	8715(%rip), %xmm8
100004f25: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
100004f2a: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
100004f2f: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004f33: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
100004f39: c5 fa 6f 64 0e 70           	vmovdqu	112(%rsi,%rcx), %xmm4
100004f3f: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100004f44: c4 e3 fd 00 6c 0e 60 4e     	vpermq	$78, 96(%rsi,%rcx), %ymm5
100004f4c: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004f52: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004f57: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
100004f5b: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100004f61: c5 fa 6f 74 0e 50           	vmovdqu	80(%rsi,%rcx), %xmm6
100004f67: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
100004f6c: c4 63 fd 00 4c 0e 40 4e     	vpermq	$78, 64(%rsi,%rcx), %ymm9
100004f74: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
100004f7a: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
100004f7f: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100004f84: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004f8a: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100004f90: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004f96: c5 79 6f 05 a2 21 00 00     	vmovdqa	8610(%rip), %xmm8
100004f9e: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100004fa3: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100004fa8: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
100004fac: c5 79 6f 1d 9c 21 00 00     	vmovdqa	8604(%rip), %xmm11
100004fb4: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
100004fb9: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100004fbe: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
100004fc2: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
100004fc8: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
100004fcd: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
100004fd2: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004fd6: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004fdc: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
100004fe1: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
100004fe6: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100004fea: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004ff0: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004ff6: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
100004ffc: c5 79 6f 1d 5c 21 00 00     	vmovdqa	8540(%rip), %xmm11
100005004: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005009: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
10000500e: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
100005012: c5 f9 6f 0d 56 21 00 00     	vmovdqa	8534(%rip), %xmm1
10000501a: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
10000501e: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100005023: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100005028: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
10000502c: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100005032: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005037: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
10000503c: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005040: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005046: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
10000504b: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100005050: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100005054: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000505a: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005060: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100005066: c5 f9 6f 0d 12 21 00 00     	vmovdqa	8466(%rip), %xmm1
10000506e: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100005073: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100005078: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000507c: c5 f9 6f 15 0c 21 00 00     	vmovdqa	8460(%rip), %xmm2
100005084: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100005088: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
10000508d: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100005092: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005096: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
10000509c: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
1000050a1: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
1000050a6: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000050aa: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000050b0: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
1000050b5: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
1000050ba: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000050be: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
1000050c4: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
1000050ca: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
1000050d0: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
1000050d5: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
1000050da: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
1000050df: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
1000050e4: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000050e9: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
1000050ed: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000050f1: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
1000050f5: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
1000050f9: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000050fd: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100005101: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005105: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100005109: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
10000510f: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
100005115: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
10000511b: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005121: c5 fe 7f 4c 0f 40           	vmovdqu	%ymm1, 64(%rdi,%rcx)
100005127: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
10000512d: c5 fe 7f 6c 0f 20           	vmovdqu	%ymm5, 32(%rdi,%rcx)
100005133: c5 fe 7f 14 0f              	vmovdqu	%ymm2, (%rdi,%rcx)
100005138: c5 7a 6f a4 0e 80 00 00 00  	vmovdqu	128(%rsi,%rcx), %xmm12
100005141: c5 7a 6f ac 0e 90 00 00 00  	vmovdqu	144(%rsi,%rcx), %xmm13
10000514a: c5 7a 6f b4 0e a0 00 00 00  	vmovdqu	160(%rsi,%rcx), %xmm14
100005153: c5 fa 6f 9c 0e b0 00 00 00  	vmovdqu	176(%rsi,%rcx), %xmm3
10000515c: c5 f9 6f 05 bc 1f 00 00     	vmovdqa	8124(%rip), %xmm0
100005164: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
100005169: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
10000516e: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100005172: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100005176: c5 f9 6f 05 b2 1f 00 00     	vmovdqa	8114(%rip), %xmm0
10000517e: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100005183: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
100005188: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
10000518c: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005190: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
100005196: c5 fa 6f a4 0e f0 00 00 00  	vmovdqu	240(%rsi,%rcx), %xmm4
10000519f: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
1000051a4: c4 e3 fd 00 ac 0e e0 00 00 00 4e    	vpermq	$78, 224(%rsi,%rcx), %ymm5
1000051af: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
1000051b5: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
1000051ba: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
1000051be: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
1000051c4: c5 fa 6f b4 0e d0 00 00 00  	vmovdqu	208(%rsi,%rcx), %xmm6
1000051cd: c4 e3 fd 00 bc 0e c0 00 00 00 4e    	vpermq	$78, 192(%rsi,%rcx), %ymm7
1000051d8: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
1000051dd: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000051e3: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
1000051e8: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
1000051ec: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000051f2: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
1000051f8: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
1000051fe: c5 79 6f 3d 3a 1f 00 00     	vmovdqa	7994(%rip), %xmm15
100005206: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
10000520b: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
100005210: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
100005214: c5 f9 6f 05 34 1f 00 00     	vmovdqa	7988(%rip), %xmm0
10000521c: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100005221: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005226: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000522a: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
100005230: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
100005235: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
10000523a: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000523e: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005244: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005249: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
10000524e: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100005252: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005258: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
10000525e: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005264: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005269: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000526e: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100005272: c5 f9 6f 05 f6 1e 00 00     	vmovdqa	7926(%rip), %xmm0
10000527a: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000527f: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005284: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005288: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
10000528e: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005293: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100005298: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000529c: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
1000052a1: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
1000052a6: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
1000052aa: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000052b0: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000052b6: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000052bc: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
1000052c2: c5 79 6f 3d b6 1e 00 00     	vmovdqa	7862(%rip), %xmm15
1000052ca: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
1000052cf: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
1000052d4: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000052d8: c5 f9 6f 05 b0 1e 00 00     	vmovdqa	7856(%rip), %xmm0
1000052e0: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
1000052e5: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
1000052ea: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000052ee: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
1000052f4: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
1000052f9: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
1000052fe: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005302: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
100005307: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
10000530c: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005310: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005316: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
10000531c: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100005322: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
100005328: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
10000532d: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
100005332: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
100005337: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
10000533c: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100005340: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005344: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005348: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
10000534c: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100005350: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100005354: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005358: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
10000535c: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100005362: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100005368: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
10000536e: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005374: c5 fe 7f 8c 0f c0 00 00 00  	vmovdqu	%ymm1, 192(%rdi,%rcx)
10000537d: c5 fe 7f 84 0f e0 00 00 00  	vmovdqu	%ymm0, 224(%rdi,%rcx)
100005386: c5 fe 7f 9c 0f a0 00 00 00  	vmovdqu	%ymm3, 160(%rdi,%rcx)
10000538f: c5 fe 7f 94 0f 80 00 00 00  	vmovdqu	%ymm2, 128(%rdi,%rcx)
100005398: 48 81 c1 00 01 00 00        	addq	$256, %rcx
10000539f: 48 83 c0 02                 	addq	$2, %rax
1000053a3: 0f 85 47 fb ff ff           	jne	-1209 <__Z4ReLUPaS_j+0x2e0>
1000053a9: 4d 85 c9                    	testq	%r9, %r9
1000053ac: 0f 84 3e 02 00 00           	je	574 <__Z4ReLUPaS_j+0x9e0>
1000053b2: c5 7a 6f 14 0e              	vmovdqu	(%rsi,%rcx), %xmm10
1000053b7: c5 7a 6f 5c 0e 10           	vmovdqu	16(%rsi,%rcx), %xmm11
1000053bd: c5 7a 6f 64 0e 20           	vmovdqu	32(%rsi,%rcx), %xmm12
1000053c3: c5 7a 6f 6c 0e 30           	vmovdqu	48(%rsi,%rcx), %xmm13
1000053c9: c5 f9 6f 35 4f 1d 00 00     	vmovdqa	7503(%rip), %xmm6
1000053d1: c4 e2 11 00 e6              	vpshufb	%xmm6, %xmm13, %xmm4
1000053d6: c4 e2 19 00 ee              	vpshufb	%xmm6, %xmm12, %xmm5
1000053db: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
1000053df: c5 f9 6f 05 49 1d 00 00     	vmovdqa	7497(%rip), %xmm0
1000053e7: c4 e2 21 00 e8              	vpshufb	%xmm0, %xmm11, %xmm5
1000053ec: c4 e2 29 00 f8              	vpshufb	%xmm0, %xmm10, %xmm7
1000053f1: c5 c1 62 ed                 	vpunpckldq	%xmm5, %xmm7, %xmm5
1000053f5: c4 63 51 02 c4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm8
1000053fb: c5 7a 6f 74 0e 70           	vmovdqu	112(%rsi,%rcx), %xmm14
100005401: c4 e2 09 00 fe              	vpshufb	%xmm6, %xmm14, %xmm7
100005406: c4 e3 fd 00 6c 0e 60 4e     	vpermq	$78, 96(%rsi,%rcx), %ymm5
10000540e: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005414: c4 e2 51 00 f6              	vpshufb	%xmm6, %xmm5, %xmm6
100005419: c5 c9 62 f7                 	vpunpckldq	%xmm7, %xmm6, %xmm6
10000541d: c4 63 7d 38 ce 01           	vinserti128	$1, %xmm6, %ymm0, %ymm9
100005423: c5 fa 6f 74 0e 50           	vmovdqu	80(%rsi,%rcx), %xmm6
100005429: c4 e2 49 00 c8              	vpshufb	%xmm0, %xmm6, %xmm1
10000542e: c4 e3 fd 00 7c 0e 40 4e     	vpermq	$78, 64(%rsi,%rcx), %ymm7
100005436: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
10000543c: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005441: c5 f9 62 c1                 	vpunpckldq	%xmm1, %xmm0, %xmm0
100005445: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000544b: c4 c3 7d 02 c1 c0           	vpblendd	$192, %ymm9, %ymm0, %ymm0
100005451: c4 63 3d 02 c0 f0           	vpblendd	$240, %ymm0, %ymm8, %ymm8
100005457: c5 f9 6f 05 e1 1c 00 00     	vmovdqa	7393(%rip), %xmm0
10000545f: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100005464: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005469: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000546d: c5 f9 6f 15 db 1c 00 00     	vmovdqa	7387(%rip), %xmm2
100005475: c4 e2 21 00 da              	vpshufb	%xmm2, %xmm11, %xmm3
10000547a: c4 e2 29 00 e2              	vpshufb	%xmm2, %xmm10, %xmm4
10000547f: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005483: c4 e3 61 02 c9 0c           	vpblendd	$12, %xmm1, %xmm3, %xmm1
100005489: c4 e2 09 00 d8              	vpshufb	%xmm0, %xmm14, %xmm3
10000548e: c4 e2 51 00 c0              	vpshufb	%xmm0, %xmm5, %xmm0
100005493: c5 f9 62 c3                 	vpunpckldq	%xmm3, %xmm0, %xmm0
100005497: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000549d: c4 e2 49 00 da              	vpshufb	%xmm2, %xmm6, %xmm3
1000054a2: c4 e2 41 00 d2              	vpshufb	%xmm2, %xmm7, %xmm2
1000054a7: c5 e9 62 d3                 	vpunpckldq	%xmm3, %xmm2, %xmm2
1000054ab: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000054b1: c4 e3 6d 02 c0 c0           	vpblendd	$192, %ymm0, %ymm2, %ymm0
1000054b7: c4 63 75 02 c8 f0           	vpblendd	$240, %ymm0, %ymm1, %ymm9
1000054bd: c5 f9 6f 05 9b 1c 00 00     	vmovdqa	7323(%rip), %xmm0
1000054c5: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000054ca: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000054cf: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000054d3: c5 f9 6f 15 95 1c 00 00     	vmovdqa	7317(%rip), %xmm2
1000054db: c4 e2 21 00 da              	vpshufb	%xmm2, %xmm11, %xmm3
1000054e0: c4 e2 29 00 e2              	vpshufb	%xmm2, %xmm10, %xmm4
1000054e5: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000054e9: c4 e3 61 02 c9 0c           	vpblendd	$12, %xmm1, %xmm3, %xmm1
1000054ef: c4 e2 09 00 d8              	vpshufb	%xmm0, %xmm14, %xmm3
1000054f4: c4 e2 51 00 c0              	vpshufb	%xmm0, %xmm5, %xmm0
1000054f9: c5 f9 62 c3                 	vpunpckldq	%xmm3, %xmm0, %xmm0
1000054fd: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005503: c4 e2 49 00 da              	vpshufb	%xmm2, %xmm6, %xmm3
100005508: c4 e2 41 00 d2              	vpshufb	%xmm2, %xmm7, %xmm2
10000550d: c5 e9 62 d3                 	vpunpckldq	%xmm3, %xmm2, %xmm2
100005511: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005517: c4 e3 6d 02 c0 c0           	vpblendd	$192, %ymm0, %ymm2, %ymm0
10000551d: c4 63 75 02 f8 f0           	vpblendd	$240, %ymm0, %ymm1, %ymm15
100005523: c5 f9 6f 0d 55 1c 00 00     	vmovdqa	7253(%rip), %xmm1
10000552b: c4 e2 11 00 d1              	vpshufb	%xmm1, %xmm13, %xmm2
100005530: c4 e2 19 00 d9              	vpshufb	%xmm1, %xmm12, %xmm3
100005535: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005539: c5 f9 6f 1d 4f 1c 00 00     	vmovdqa	7247(%rip), %xmm3
100005541: c4 e2 21 00 e3              	vpshufb	%xmm3, %xmm11, %xmm4
100005546: c4 e2 29 00 c3              	vpshufb	%xmm3, %xmm10, %xmm0
10000554b: c5 f9 62 c4                 	vpunpckldq	%xmm4, %xmm0, %xmm0
10000554f: c4 e3 79 02 c2 0c           	vpblendd	$12, %xmm2, %xmm0, %xmm0
100005555: c4 e2 09 00 d1              	vpshufb	%xmm1, %xmm14, %xmm2
10000555a: c4 e2 51 00 c9              	vpshufb	%xmm1, %xmm5, %xmm1
10000555f: c5 f1 62 ca                 	vpunpckldq	%xmm2, %xmm1, %xmm1
100005563: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005569: c4 e2 49 00 d3              	vpshufb	%xmm3, %xmm6, %xmm2
10000556e: c4 e2 41 00 db              	vpshufb	%xmm3, %xmm7, %xmm3
100005573: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005577: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
10000557d: c4 e3 6d 02 c9 c0           	vpblendd	$192, %ymm1, %ymm2, %ymm1
100005583: c4 e3 7d 02 c1 f0           	vpblendd	$240, %ymm1, %ymm0, %ymm0
100005589: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
10000558d: c4 e2 3d 3c d1              	vpmaxsb	%ymm1, %ymm8, %ymm2
100005592: c4 e2 35 3c d9              	vpmaxsb	%ymm1, %ymm9, %ymm3
100005597: c4 e2 05 3c e1              	vpmaxsb	%ymm1, %ymm15, %ymm4
10000559c: c4 e2 7d 3c c1              	vpmaxsb	%ymm1, %ymm0, %ymm0
1000055a1: c5 ed 60 cb                 	vpunpcklbw	%ymm3, %ymm2, %ymm1
1000055a5: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000055a9: c5 dd 60 d8                 	vpunpcklbw	%ymm0, %ymm4, %ymm3
1000055ad: c5 dd 68 c0                 	vpunpckhbw	%ymm0, %ymm4, %ymm0
1000055b1: c5 f5 61 e3                 	vpunpcklwd	%ymm3, %ymm1, %ymm4
1000055b5: c5 f5 69 cb                 	vpunpckhwd	%ymm3, %ymm1, %ymm1
1000055b9: c5 ed 61 d8                 	vpunpcklwd	%ymm0, %ymm2, %ymm3
1000055bd: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000055c1: c4 e3 5d 38 d1 01           	vinserti128	$1, %xmm1, %ymm4, %ymm2
1000055c7: c4 e3 65 38 e8 01           	vinserti128	$1, %xmm0, %ymm3, %ymm5
1000055cd: c4 e3 5d 46 c9 31           	vperm2i128	$49, %ymm1, %ymm4, %ymm1
1000055d3: c4 e3 65 46 c0 31           	vperm2i128	$49, %ymm0, %ymm3, %ymm0
1000055d9: c5 fe 7f 4c 0f 40           	vmovdqu	%ymm1, 64(%rdi,%rcx)
1000055df: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
1000055e5: c5 fe 7f 6c 0f 20           	vmovdqu	%ymm5, 32(%rdi,%rcx)
1000055eb: c5 fe 7f 14 0f              	vmovdqu	%ymm2, (%rdi,%rcx)
1000055f0: 4a 8d 34 86                 	leaq	(%rsi,%r8,4), %rsi
1000055f4: 4a 8d 3c 87                 	leaq	(%rdi,%r8,4), %rdi
1000055f8: 4d 39 d0                    	cmpq	%r10, %r8
1000055fb: 0f 84 a1 f6 ff ff           	je	-2399 <__Z4ReLUPaS_j+0x92>
100005601: 41 c1 e0 02                 	shll	$2, %r8d
100005605: 89 d0                       	movl	%edx, %eax
100005607: 44 29 c0                    	subl	%r8d, %eax
10000560a: e9 47 f6 ff ff              	jmp	-2489 <__Z4ReLUPaS_j+0x46>
10000560f: 31 c9                       	xorl	%ecx, %ecx
100005611: 4d 85 c9                    	testq	%r9, %r9
100005614: 0f 85 8a f7 ff ff           	jne	-2166 <__Z4ReLUPaS_j+0x194>
10000561a: e9 bb f7 ff ff              	jmp	-2117 <__Z4ReLUPaS_j+0x1ca>
10000561f: 31 c9                       	xorl	%ecx, %ecx
100005621: 4d 85 c9                    	testq	%r9, %r9
100005624: 0f 85 88 fd ff ff           	jne	-632 <__Z4ReLUPaS_j+0x7a2>
10000562a: eb c4                       	jmp	-60 <__Z4ReLUPaS_j+0x9e0>
10000562c: 90                          	nop
10000562d: 90                          	nop
10000562e: 90                          	nop
10000562f: 90                          	nop

0000000100005630 __ZN11LineNetworkC2Ev:
100005630: 55                          	pushq	%rbp
100005631: 48 89 e5                    	movq	%rsp, %rbp
100005634: 41 56                       	pushq	%r14
100005636: 53                          	pushq	%rbx
100005637: 48 89 fb                    	movq	%rdi, %rbx
10000563a: e8 f1 f2 ff ff              	callq	-3343 <__ZN14ModelInterfaceC2Ev>
10000563f: 48 8d 05 aa 3a 00 00        	leaq	15018(%rip), %rax
100005646: 48 89 03                    	movq	%rax, (%rbx)
100005649: 48 89 df                    	movq	%rbx, %rdi
10000564c: be 00 00 08 00              	movl	$524288, %esi
100005651: e8 1a f4 ff ff              	callq	-3046 <__ZN14ModelInterface11init_bufferEj>
100005656: c7 43 20 00 04 a2 01        	movl	$27395072, 32(%rbx)
10000565d: c5 f8 28 05 3b 1b 00 00     	vmovaps	6971(%rip), %xmm0
100005665: c5 f8 11 43 08              	vmovups	%xmm0, 8(%rbx)
10000566a: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
100005674: 48 89 43 18                 	movq	%rax, 24(%rbx)
100005678: 5b                          	popq	%rbx
100005679: 41 5e                       	popq	%r14
10000567b: 5d                          	popq	%rbp
10000567c: c3                          	retq
10000567d: 49 89 c6                    	movq	%rax, %r14
100005680: 48 89 df                    	movq	%rbx, %rdi
100005683: e8 e8 f2 ff ff              	callq	-3352 <__ZN14ModelInterfaceD2Ev>
100005688: 4c 89 f7                    	movq	%r14, %rdi
10000568b: e8 48 17 00 00              	callq	5960 <dyld_stub_binder+0x100006dd8>
100005690: 0f 0b                       	ud2
100005692: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000569c: 0f 1f 40 00                 	nopl	(%rax)

00000001000056a0 __ZN11LineNetworkC1Ev:
1000056a0: 55                          	pushq	%rbp
1000056a1: 48 89 e5                    	movq	%rsp, %rbp
1000056a4: 41 56                       	pushq	%r14
1000056a6: 53                          	pushq	%rbx
1000056a7: 48 89 fb                    	movq	%rdi, %rbx
1000056aa: e8 81 f2 ff ff              	callq	-3455 <__ZN14ModelInterfaceC2Ev>
1000056af: 48 8d 05 3a 3a 00 00        	leaq	14906(%rip), %rax
1000056b6: 48 89 03                    	movq	%rax, (%rbx)
1000056b9: 48 89 df                    	movq	%rbx, %rdi
1000056bc: be 00 00 08 00              	movl	$524288, %esi
1000056c1: e8 aa f3 ff ff              	callq	-3158 <__ZN14ModelInterface11init_bufferEj>
1000056c6: c7 43 20 00 04 a2 01        	movl	$27395072, 32(%rbx)
1000056cd: c5 f8 28 05 cb 1a 00 00     	vmovaps	6859(%rip), %xmm0
1000056d5: c5 f8 11 43 08              	vmovups	%xmm0, 8(%rbx)
1000056da: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
1000056e4: 48 89 43 18                 	movq	%rax, 24(%rbx)
1000056e8: 5b                          	popq	%rbx
1000056e9: 41 5e                       	popq	%r14
1000056eb: 5d                          	popq	%rbp
1000056ec: c3                          	retq
1000056ed: 49 89 c6                    	movq	%rax, %r14
1000056f0: 48 89 df                    	movq	%rbx, %rdi
1000056f3: e8 78 f2 ff ff              	callq	-3464 <__ZN14ModelInterfaceD2Ev>
1000056f8: 4c 89 f7                    	movq	%r14, %rdi
1000056fb: e8 d8 16 00 00              	callq	5848 <dyld_stub_binder+0x100006dd8>
100005700: 0f 0b                       	ud2
100005702: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000570c: 0f 1f 40 00                 	nopl	(%rax)

0000000100005710 __ZN11LineNetwork7forwardEv:
100005710: 55                          	pushq	%rbp
100005711: 48 89 e5                    	movq	%rsp, %rbp
100005714: 41 57                       	pushq	%r15
100005716: 41 56                       	pushq	%r14
100005718: 41 55                       	pushq	%r13
10000571a: 41 54                       	pushq	%r12
10000571c: 53                          	pushq	%rbx
10000571d: 48 83 ec 58                 	subq	$88, %rsp
100005721: 49 89 ff                    	movq	%rdi, %r15
100005724: e8 27 f3 ff ff              	callq	-3289 <__ZN14ModelInterface13output_bufferEv>
100005729: 49 89 c6                    	movq	%rax, %r14
10000572c: 4c 89 ff                    	movq	%r15, %rdi
10000572f: e8 0c f3 ff ff              	callq	-3316 <__ZN14ModelInterface12input_bufferEv>
100005734: 48 8d 15 25 1c 00 00        	leaq	7205(%rip), %rdx
10000573b: 48 8d 0d 66 1c 00 00        	leaq	7270(%rip), %rcx
100005742: 4c 89 f7                    	movq	%r14, %rdi
100005745: 48 89 c6                    	movq	%rax, %rsi
100005748: 41 b8 37 00 00 00           	movl	$55, %r8d
10000574e: e8 6d 05 00 00              	callq	1389 <__ZN11LineNetwork7forwardEv+0x5b0>
100005753: 4c 89 ff                    	movq	%r15, %rdi
100005756: e8 a5 f4 ff ff              	callq	-2907 <__ZN14ModelInterface11swap_bufferEv>
10000575b: 4c 89 ff                    	movq	%r15, %rdi
10000575e: e8 ed f2 ff ff              	callq	-3347 <__ZN14ModelInterface13output_bufferEv>
100005763: 49 89 c6                    	movq	%rax, %r14
100005766: 4c 89 ff                    	movq	%r15, %rdi
100005769: e8 d2 f2 ff ff              	callq	-3374 <__ZN14ModelInterface12input_bufferEv>
10000576e: 4c 89 f7                    	movq	%r14, %rdi
100005771: 48 89 c6                    	movq	%rax, %rsi
100005774: ba 00 00 08 00              	movl	$524288, %edx
100005779: e8 92 f4 ff ff              	callq	-2926 <__Z4ReLUPaS_j>
10000577e: 4c 89 ff                    	movq	%r15, %rdi
100005781: e8 7a f4 ff ff              	callq	-2950 <__ZN14ModelInterface11swap_bufferEv>
100005786: 4c 89 ff                    	movq	%r15, %rdi
100005789: e8 c2 f2 ff ff              	callq	-3390 <__ZN14ModelInterface13output_bufferEv>
10000578e: 49 89 c5                    	movq	%rax, %r13
100005791: 4c 89 7d 88                 	movq	%r15, -120(%rbp)
100005795: 4c 89 ff                    	movq	%r15, %rdi
100005798: e8 a3 f2 ff ff              	callq	-3421 <__ZN14ModelInterface12input_bufferEv>
10000579d: 48 89 45 c8                 	movq	%rax, -56(%rbp)
1000057a1: 31 c0                       	xorl	%eax, %eax
1000057a3: eb 23                       	jmp	35 <__ZN11LineNetwork7forwardEv+0xb8>
1000057a5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000057af: 90                          	nop
1000057b0: 48 8b 45 c0                 	movq	-64(%rbp), %rax
1000057b4: 48 ff c0                    	incq	%rax
1000057b7: 4c 8b 6d b8                 	movq	-72(%rbp), %r13
1000057bb: 49 ff c5                    	incq	%r13
1000057be: 48 83 f8 08                 	cmpq	$8, %rax
1000057c2: 0f 84 02 01 00 00           	je	258 <__ZN11LineNetwork7forwardEv+0x1ba>
1000057c8: 48 89 45 c0                 	movq	%rax, -64(%rbp)
1000057cc: 48 8d 04 c5 00 00 00 00     	leaq	(,%rax,8), %rax
1000057d4: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
1000057d8: 48 8d 05 d1 1b 00 00        	leaq	7121(%rip), %rax
1000057df: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
1000057e3: 48 89 55 90                 	movq	%rdx, -112(%rbp)
1000057e7: 48 8d 54 08 18              	leaq	24(%rax,%rcx), %rdx
1000057ec: 48 89 55 98                 	movq	%rdx, -104(%rbp)
1000057f0: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
1000057f4: 48 8d 44 08 30              	leaq	48(%rax,%rcx), %rax
1000057f9: 48 89 45 a0                 	movq	%rax, -96(%rbp)
1000057fd: 4c 89 6d b8                 	movq	%r13, -72(%rbp)
100005801: 4c 8b 7d c8                 	movq	-56(%rbp), %r15
100005805: 31 c0                       	xorl	%eax, %eax
100005807: eb 25                       	jmp	37 <__ZN11LineNetwork7forwardEv+0x11e>
100005809: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005810: 4c 8b 7d a8                 	movq	-88(%rbp), %r15
100005814: 49 81 c7 00 10 00 00        	addq	$4096, %r15
10000581b: 49 81 c5 00 04 00 00        	addq	$1024, %r13
100005822: 48 8b 45 b0                 	movq	-80(%rbp), %rax
100005826: 48 3d fd 00 00 00           	cmpq	$253, %rax
10000582c: 73 82                       	jae	-126 <__ZN11LineNetwork7forwardEv+0xa0>
10000582e: 48 83 c0 02                 	addq	$2, %rax
100005832: 48 89 45 b0                 	movq	%rax, -80(%rbp)
100005836: 4c 89 7d a8                 	movq	%r15, -88(%rbp)
10000583a: 31 db                       	xorl	%ebx, %ebx
10000583c: eb 18                       	jmp	24 <__ZN11LineNetwork7forwardEv+0x146>
10000583e: 66 90                       	nop
100005840: 41 88 44 9d 00              	movb	%al, (%r13,%rbx,4)
100005845: 48 83 c3 02                 	addq	$2, %rbx
100005849: 49 83 c7 10                 	addq	$16, %r15
10000584d: 48 81 fb fd 00 00 00        	cmpq	$253, %rbx
100005854: 73 ba                       	jae	-70 <__ZN11LineNetwork7forwardEv+0x100>
100005856: 4c 89 ff                    	movq	%r15, %rdi
100005859: 48 8b 75 90                 	movq	-112(%rbp), %rsi
10000585d: e8 1e 12 00 00              	callq	4638 <__ZN11LineNetwork7forwardEv+0x1370>
100005862: 41 89 c6                    	movl	%eax, %r14d
100005865: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
10000586c: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100005870: e8 0b 12 00 00              	callq	4619 <__ZN11LineNetwork7forwardEv+0x1370>
100005875: 41 89 c4                    	movl	%eax, %r12d
100005878: 45 01 f4                    	addl	%r14d, %r12d
10000587b: 49 8d bf 00 10 00 00        	leaq	4096(%r15), %rdi
100005882: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005886: e8 f5 11 00 00              	callq	4597 <__ZN11LineNetwork7forwardEv+0x1370>
10000588b: 44 01 e0                    	addl	%r12d, %eax
10000588e: 48 8d 0d 5b 1d 00 00        	leaq	7515(%rip), %rcx
100005895: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
100005899: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
10000589d: 01 c1                       	addl	%eax, %ecx
10000589f: 6b c9 37                    	imull	$55, %ecx, %ecx
1000058a2: 89 c8                       	movl	%ecx, %eax
1000058a4: c1 f8 1f                    	sarl	$31, %eax
1000058a7: c1 e8 12                    	shrl	$18, %eax
1000058aa: 01 c8                       	addl	%ecx, %eax
1000058ac: c1 f8 0e                    	sarl	$14, %eax
1000058af: 3d 80 00 00 00              	cmpl	$128, %eax
1000058b4: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x1ab>
1000058b6: b8 7f 00 00 00              	movl	$127, %eax
1000058bb: 83 f8 81                    	cmpl	$-127, %eax
1000058be: 7f 80                       	jg	-128 <__ZN11LineNetwork7forwardEv+0x130>
1000058c0: b8 81 00 00 00              	movl	$129, %eax
1000058c5: e9 76 ff ff ff              	jmp	-138 <__ZN11LineNetwork7forwardEv+0x130>
1000058ca: 4c 8b 7d 88                 	movq	-120(%rbp), %r15
1000058ce: 4c 89 ff                    	movq	%r15, %rdi
1000058d1: e8 2a f3 ff ff              	callq	-3286 <__ZN14ModelInterface11swap_bufferEv>
1000058d6: 4c 89 ff                    	movq	%r15, %rdi
1000058d9: e8 72 f1 ff ff              	callq	-3726 <__ZN14ModelInterface13output_bufferEv>
1000058de: 49 89 c6                    	movq	%rax, %r14
1000058e1: 4c 89 ff                    	movq	%r15, %rdi
1000058e4: e8 57 f1 ff ff              	callq	-3753 <__ZN14ModelInterface12input_bufferEv>
1000058e9: 4c 89 f7                    	movq	%r14, %rdi
1000058ec: 48 89 c6                    	movq	%rax, %rsi
1000058ef: ba 00 00 02 00              	movl	$131072, %edx
1000058f4: e8 17 f3 ff ff              	callq	-3305 <__Z4ReLUPaS_j>
1000058f9: 4c 89 ff                    	movq	%r15, %rdi
1000058fc: e8 ff f2 ff ff              	callq	-3329 <__ZN14ModelInterface11swap_bufferEv>
100005901: 4c 89 ff                    	movq	%r15, %rdi
100005904: e8 47 f1 ff ff              	callq	-3769 <__ZN14ModelInterface13output_bufferEv>
100005909: 49 89 c5                    	movq	%rax, %r13
10000590c: 4c 89 ff                    	movq	%r15, %rdi
10000590f: e8 2c f1 ff ff              	callq	-3796 <__ZN14ModelInterface12input_bufferEv>
100005914: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100005918: 31 c0                       	xorl	%eax, %eax
10000591a: eb 1c                       	jmp	28 <__ZN11LineNetwork7forwardEv+0x228>
10000591c: 0f 1f 40 00                 	nopl	(%rax)
100005920: 48 8b 45 c0                 	movq	-64(%rbp), %rax
100005924: 48 ff c0                    	incq	%rax
100005927: 4c 8b 6d b8                 	movq	-72(%rbp), %r13
10000592b: 49 ff c5                    	incq	%r13
10000592e: 48 83 f8 10                 	cmpq	$16, %rax
100005932: 0f 84 ff 00 00 00           	je	255 <__ZN11LineNetwork7forwardEv+0x327>
100005938: 48 89 45 c0                 	movq	%rax, -64(%rbp)
10000593c: 48 8d 04 c5 00 00 00 00     	leaq	(,%rax,8), %rax
100005944: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005948: 48 8d 05 b1 1c 00 00        	leaq	7345(%rip), %rax
10000594f: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005953: 48 89 55 90                 	movq	%rdx, -112(%rbp)
100005957: 48 8d 54 08 18              	leaq	24(%rax,%rcx), %rdx
10000595c: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005960: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
100005964: 48 8d 44 08 30              	leaq	48(%rax,%rcx), %rax
100005969: 48 89 45 a0                 	movq	%rax, -96(%rbp)
10000596d: 4c 89 6d b8                 	movq	%r13, -72(%rbp)
100005971: 4c 8b 7d c8                 	movq	-56(%rbp), %r15
100005975: 31 c0                       	xorl	%eax, %eax
100005977: eb 23                       	jmp	35 <__ZN11LineNetwork7forwardEv+0x28c>
100005979: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005980: 4c 8b 7d a8                 	movq	-88(%rbp), %r15
100005984: 49 81 c7 00 08 00 00        	addq	$2048, %r15
10000598b: 49 81 c5 00 04 00 00        	addq	$1024, %r13
100005992: 48 8b 45 b0                 	movq	-80(%rbp), %rax
100005996: 48 83 f8 7d                 	cmpq	$125, %rax
10000599a: 73 84                       	jae	-124 <__ZN11LineNetwork7forwardEv+0x210>
10000599c: 48 83 c0 02                 	addq	$2, %rax
1000059a0: 48 89 45 b0                 	movq	%rax, -80(%rbp)
1000059a4: 4c 89 7d a8                 	movq	%r15, -88(%rbp)
1000059a8: 31 db                       	xorl	%ebx, %ebx
1000059aa: eb 17                       	jmp	23 <__ZN11LineNetwork7forwardEv+0x2b3>
1000059ac: 0f 1f 40 00                 	nopl	(%rax)
1000059b0: 41 88 44 dd 00              	movb	%al, (%r13,%rbx,8)
1000059b5: 48 83 c3 02                 	addq	$2, %rbx
1000059b9: 49 83 c7 10                 	addq	$16, %r15
1000059bd: 48 83 fb 7d                 	cmpq	$125, %rbx
1000059c1: 73 bd                       	jae	-67 <__ZN11LineNetwork7forwardEv+0x270>
1000059c3: 4c 89 ff                    	movq	%r15, %rdi
1000059c6: 48 8b 75 90                 	movq	-112(%rbp), %rsi
1000059ca: e8 b1 10 00 00              	callq	4273 <__ZN11LineNetwork7forwardEv+0x1370>
1000059cf: 41 89 c6                    	movl	%eax, %r14d
1000059d2: 49 8d bf 00 04 00 00        	leaq	1024(%r15), %rdi
1000059d9: 48 8b 75 98                 	movq	-104(%rbp), %rsi
1000059dd: e8 9e 10 00 00              	callq	4254 <__ZN11LineNetwork7forwardEv+0x1370>
1000059e2: 41 89 c4                    	movl	%eax, %r12d
1000059e5: 45 01 f4                    	addl	%r14d, %r12d
1000059e8: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
1000059ef: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
1000059f3: e8 88 10 00 00              	callq	4232 <__ZN11LineNetwork7forwardEv+0x1370>
1000059f8: 44 01 e0                    	addl	%r12d, %eax
1000059fb: 48 8d 0d 7e 20 00 00        	leaq	8318(%rip), %rcx
100005a02: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
100005a06: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
100005a0a: 01 c1                       	addl	%eax, %ecx
100005a0c: 6b c9 39                    	imull	$57, %ecx, %ecx
100005a0f: 89 c8                       	movl	%ecx, %eax
100005a11: c1 f8 1f                    	sarl	$31, %eax
100005a14: c1 e8 12                    	shrl	$18, %eax
100005a17: 01 c8                       	addl	%ecx, %eax
100005a19: c1 f8 0e                    	sarl	$14, %eax
100005a1c: 3d 80 00 00 00              	cmpl	$128, %eax
100005a21: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x318>
100005a23: b8 7f 00 00 00              	movl	$127, %eax
100005a28: 83 f8 81                    	cmpl	$-127, %eax
100005a2b: 7f 83                       	jg	-125 <__ZN11LineNetwork7forwardEv+0x2a0>
100005a2d: b8 81 00 00 00              	movl	$129, %eax
100005a32: e9 79 ff ff ff              	jmp	-135 <__ZN11LineNetwork7forwardEv+0x2a0>
100005a37: 4c 8b 7d 88                 	movq	-120(%rbp), %r15
100005a3b: 4c 89 ff                    	movq	%r15, %rdi
100005a3e: e8 bd f1 ff ff              	callq	-3651 <__ZN14ModelInterface11swap_bufferEv>
100005a43: 4c 89 ff                    	movq	%r15, %rdi
100005a46: e8 05 f0 ff ff              	callq	-4091 <__ZN14ModelInterface13output_bufferEv>
100005a4b: 49 89 c6                    	movq	%rax, %r14
100005a4e: 4c 89 ff                    	movq	%r15, %rdi
100005a51: e8 ea ef ff ff              	callq	-4118 <__ZN14ModelInterface12input_bufferEv>
100005a56: 4c 89 f7                    	movq	%r14, %rdi
100005a59: 48 89 c6                    	movq	%rax, %rsi
100005a5c: ba 00 00 01 00              	movl	$65536, %edx
100005a61: e8 aa f1 ff ff              	callq	-3670 <__Z4ReLUPaS_j>
100005a66: 4c 89 ff                    	movq	%r15, %rdi
100005a69: e8 92 f1 ff ff              	callq	-3694 <__ZN14ModelInterface11swap_bufferEv>
100005a6e: 4c 89 ff                    	movq	%r15, %rdi
100005a71: e8 da ef ff ff              	callq	-4134 <__ZN14ModelInterface13output_bufferEv>
100005a76: 48 89 c3                    	movq	%rax, %rbx
100005a79: 4c 89 ff                    	movq	%r15, %rdi
100005a7c: e8 bf ef ff ff              	callq	-4161 <__ZN14ModelInterface12input_bufferEv>
100005a81: 48 89 45 80                 	movq	%rax, -128(%rbp)
100005a85: 31 c0                       	xorl	%eax, %eax
100005a87: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x398>
100005a89: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005a90: 48 8b 45 c8                 	movq	-56(%rbp), %rax
100005a94: 48 ff c0                    	incq	%rax
100005a97: 48 8b 5d c0                 	movq	-64(%rbp), %rbx
100005a9b: 48 ff c3                    	incq	%rbx
100005a9e: 48 83 f8 20                 	cmpq	$32, %rax
100005aa2: 0f 84 17 01 00 00           	je	279 <__ZN11LineNetwork7forwardEv+0x4af>
100005aa8: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100005aac: 48 c1 e0 04                 	shlq	$4, %rax
100005ab0: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005ab4: 48 8d 05 d5 1f 00 00        	leaq	8149(%rip), %rax
100005abb: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005abf: 48 89 55 90                 	movq	%rdx, -112(%rbp)
100005ac3: 48 8d 54 08 30              	leaq	48(%rax,%rcx), %rdx
100005ac8: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005acc: 48 89 4d d0                 	movq	%rcx, -48(%rbp)
100005ad0: 48 8d 44 08 60              	leaq	96(%rax,%rcx), %rax
100005ad5: 48 89 45 a0                 	movq	%rax, -96(%rbp)
100005ad9: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
100005add: 4c 8b 7d 80                 	movq	-128(%rbp), %r15
100005ae1: 31 c0                       	xorl	%eax, %eax
100005ae3: eb 2b                       	jmp	43 <__ZN11LineNetwork7forwardEv+0x400>
100005ae5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005aef: 90                          	nop
100005af0: 4c 8b 7d b0                 	movq	-80(%rbp), %r15
100005af4: 49 81 c7 00 08 00 00        	addq	$2048, %r15
100005afb: 48 8b 5d a8                 	movq	-88(%rbp), %rbx
100005aff: 48 81 c3 00 04 00 00        	addq	$1024, %rbx
100005b06: 48 8b 45 b8                 	movq	-72(%rbp), %rax
100005b0a: 48 83 f8 3d                 	cmpq	$61, %rax
100005b0e: 73 80                       	jae	-128 <__ZN11LineNetwork7forwardEv+0x380>
100005b10: 48 83 c0 02                 	addq	$2, %rax
100005b14: 48 89 45 b8                 	movq	%rax, -72(%rbp)
100005b18: 48 89 5d a8                 	movq	%rbx, -88(%rbp)
100005b1c: 4c 89 7d b0                 	movq	%r15, -80(%rbp)
100005b20: 45 31 f6                    	xorl	%r14d, %r14d
100005b23: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x434>
100005b25: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005b2f: 90                          	nop
100005b30: 88 03                       	movb	%al, (%rbx)
100005b32: 49 83 c6 02                 	addq	$2, %r14
100005b36: 49 83 c7 20                 	addq	$32, %r15
100005b3a: 48 83 c3 20                 	addq	$32, %rbx
100005b3e: 49 83 fe 3d                 	cmpq	$61, %r14
100005b42: 73 ac                       	jae	-84 <__ZN11LineNetwork7forwardEv+0x3e0>
100005b44: 4c 89 ff                    	movq	%r15, %rdi
100005b47: 48 8b 75 90                 	movq	-112(%rbp), %rsi
100005b4b: e8 80 10 00 00              	callq	4224 <__ZN11LineNetwork7forwardEv+0x14c0>
100005b50: 41 89 c4                    	movl	%eax, %r12d
100005b53: 49 8d bf 00 04 00 00        	leaq	1024(%r15), %rdi
100005b5a: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100005b5e: e8 6d 10 00 00              	callq	4205 <__ZN11LineNetwork7forwardEv+0x14c0>
100005b63: 41 89 c5                    	movl	%eax, %r13d
100005b66: 45 01 e5                    	addl	%r12d, %r13d
100005b69: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
100005b70: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005b74: e8 57 10 00 00              	callq	4183 <__ZN11LineNetwork7forwardEv+0x14c0>
100005b79: 44 01 e8                    	addl	%r13d, %eax
100005b7c: 48 8d 0d 0d 31 00 00        	leaq	12557(%rip), %rcx
100005b83: 48 8b 55 d0                 	movq	-48(%rbp), %rdx
100005b87: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
100005b8b: 01 c1                       	addl	%eax, %ecx
100005b8d: c1 e1 04                    	shll	$4, %ecx
100005b90: 8d 0c 49                    	leal	(%rcx,%rcx,2), %ecx
100005b93: 89 c8                       	movl	%ecx, %eax
100005b95: c1 f8 1f                    	sarl	$31, %eax
100005b98: c1 e8 12                    	shrl	$18, %eax
100005b9b: 01 c8                       	addl	%ecx, %eax
100005b9d: c1 f8 0e                    	sarl	$14, %eax
100005ba0: 3d 80 00 00 00              	cmpl	$128, %eax
100005ba5: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x49c>
100005ba7: b8 7f 00 00 00              	movl	$127, %eax
100005bac: 83 f8 81                    	cmpl	$-127, %eax
100005baf: 0f 8f 7b ff ff ff           	jg	-133 <__ZN11LineNetwork7forwardEv+0x420>
100005bb5: b8 81 00 00 00              	movl	$129, %eax
100005bba: e9 71 ff ff ff              	jmp	-143 <__ZN11LineNetwork7forwardEv+0x420>
100005bbf: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100005bc3: 48 89 df                    	movq	%rbx, %rdi
100005bc6: e8 35 f0 ff ff              	callq	-4043 <__ZN14ModelInterface11swap_bufferEv>
100005bcb: 48 89 df                    	movq	%rbx, %rdi
100005bce: e8 7d ee ff ff              	callq	-4483 <__ZN14ModelInterface13output_bufferEv>
100005bd3: 49 89 c6                    	movq	%rax, %r14
100005bd6: 48 89 df                    	movq	%rbx, %rdi
100005bd9: e8 62 ee ff ff              	callq	-4510 <__ZN14ModelInterface12input_bufferEv>
100005bde: 4c 89 f7                    	movq	%r14, %rdi
100005be1: 48 89 c6                    	movq	%rax, %rsi
100005be4: ba 00 80 00 00              	movl	$32768, %edx
100005be9: e8 22 f0 ff ff              	callq	-4062 <__Z4ReLUPaS_j>
100005bee: 48 89 df                    	movq	%rbx, %rdi
100005bf1: e8 0a f0 ff ff              	callq	-4086 <__ZN14ModelInterface11swap_bufferEv>
100005bf6: 48 89 df                    	movq	%rbx, %rdi
100005bf9: e8 52 ee ff ff              	callq	-4526 <__ZN14ModelInterface13output_bufferEv>
100005bfe: 49 89 c4                    	movq	%rax, %r12
100005c01: 48 89 df                    	movq	%rbx, %rdi
100005c04: e8 37 ee ff ff              	callq	-4553 <__ZN14ModelInterface12input_bufferEv>
100005c09: 49 89 c6                    	movq	%rax, %r14
100005c0c: 31 c0                       	xorl	%eax, %eax
100005c0e: 4c 8d 3d 9b 30 00 00        	leaq	12443(%rip), %r15
100005c15: eb 21                       	jmp	33 <__ZN11LineNetwork7forwardEv+0x528>
100005c17: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100005c20: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100005c24: 48 ff c0                    	incq	%rax
100005c27: 49 83 c4 20                 	addq	$32, %r12
100005c2b: 49 81 c6 00 04 00 00        	addq	$1024, %r14
100005c32: 48 83 f8 20                 	cmpq	$32, %rax
100005c36: 74 63                       	je	99 <__ZN11LineNetwork7forwardEv+0x58b>
100005c38: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100005c3c: 4c 89 f3                    	movq	%r14, %rbx
100005c3f: 45 31 ed                    	xorl	%r13d, %r13d
100005c42: eb 1d                       	jmp	29 <__ZN11LineNetwork7forwardEv+0x551>
100005c44: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005c4e: 66 90                       	nop
100005c50: 43 88 04 2c                 	movb	%al, (%r12,%r13)
100005c54: 49 ff c5                    	incq	%r13
100005c57: 48 83 c3 20                 	addq	$32, %rbx
100005c5b: 49 83 fd 20                 	cmpq	$32, %r13
100005c5f: 74 bf                       	je	-65 <__ZN11LineNetwork7forwardEv+0x510>
100005c61: 48 89 df                    	movq	%rbx, %rdi
100005c64: 4c 89 fe                    	movq	%r15, %rsi
100005c67: e8 f4 10 00 00              	callq	4340 <__ZN11LineNetwork7forwardEv+0x1650>
100005c6c: c1 e0 05                    	shll	$5, %eax
100005c6f: 89 c1                       	movl	%eax, %ecx
100005c71: 83 c1 20                    	addl	$32, %ecx
100005c74: c1 f9 1f                    	sarl	$31, %ecx
100005c77: c1 e9 12                    	shrl	$18, %ecx
100005c7a: 8d 04 08                    	leal	(%rax,%rcx), %eax
100005c7d: 83 c0 20                    	addl	$32, %eax
100005c80: c1 f8 0e                    	sarl	$14, %eax
100005c83: 3d 80 00 00 00              	cmpl	$128, %eax
100005c88: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x57f>
100005c8a: b8 7f 00 00 00              	movl	$127, %eax
100005c8f: 83 f8 81                    	cmpl	$-127, %eax
100005c92: 7f bc                       	jg	-68 <__ZN11LineNetwork7forwardEv+0x540>
100005c94: b8 81 00 00 00              	movl	$129, %eax
100005c99: eb b5                       	jmp	-75 <__ZN11LineNetwork7forwardEv+0x540>
100005c9b: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100005c9f: 48 89 df                    	movq	%rbx, %rdi
100005ca2: e8 59 ef ff ff              	callq	-4263 <__ZN14ModelInterface11swap_bufferEv>
100005ca7: 48 89 df                    	movq	%rbx, %rdi
100005caa: 48 83 c4 58                 	addq	$88, %rsp
100005cae: 5b                          	popq	%rbx
100005caf: 41 5c                       	popq	%r12
100005cb1: 41 5d                       	popq	%r13
100005cb3: 41 5e                       	popq	%r14
100005cb5: 41 5f                       	popq	%r15
100005cb7: 5d                          	popq	%rbp
100005cb8: e9 43 ef ff ff              	jmp	-4285 <__ZN14ModelInterface11swap_bufferEv>
100005cbd: 0f 1f 00                    	nopl	(%rax)
100005cc0: 55                          	pushq	%rbp
100005cc1: 48 89 e5                    	movq	%rsp, %rbp
100005cc4: 41 57                       	pushq	%r15
100005cc6: 41 56                       	pushq	%r14
100005cc8: 41 55                       	pushq	%r13
100005cca: 41 54                       	pushq	%r12
100005ccc: 53                          	pushq	%rbx
100005ccd: 48 83 e4 e0                 	andq	$-32, %rsp
100005cd1: 48 81 ec e0 02 00 00        	subq	$736, %rsp
100005cd8: 48 89 4c 24 50              	movq	%rcx, 80(%rsp)
100005cdd: 48 89 54 24 48              	movq	%rdx, 72(%rsp)
100005ce2: 49 89 ff                    	movq	%rdi, %r15
100005ce5: c4 c1 79 6e c0              	vmovd	%r8d, %xmm0
100005cea: c4 e2 7d 58 c8              	vpbroadcastd	%xmm0, %ymm1
100005cef: 48 8d 86 01 04 00 00        	leaq	1025(%rsi), %rax
100005cf6: 48 89 44 24 40              	movq	%rax, 64(%rsp)
100005cfb: 48 8d 86 02 04 00 00        	leaq	1026(%rsi), %rax
100005d02: 48 89 44 24 38              	movq	%rax, 56(%rsp)
100005d07: 45 31 c9                    	xorl	%r9d, %r9d
100005d0a: c5 fd 6f 15 ce 15 00 00     	vmovdqa	5582(%rip), %ymm2
100005d12: 44 89 44 24 14              	movl	%r8d, 20(%rsp)
100005d17: 48 89 74 24 58              	movq	%rsi, 88(%rsp)
100005d1c: c5 fd 7f 8c 24 60 02 00 00  	vmovdqa	%ymm1, 608(%rsp)
100005d25: eb 19                       	jmp	25 <__ZN11LineNetwork7forwardEv+0x630>
100005d27: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100005d30: 49 ff c1                    	incq	%r9
100005d33: 48 ff c7                    	incq	%rdi
100005d36: 49 83 f9 08                 	cmpq	$8, %r9
100005d3a: 0f 84 f2 0c 00 00           	je	3314 <__ZN11LineNetwork7forwardEv+0x1322>
100005d40: 49 8d 81 f1 07 00 00        	leaq	2033(%r9), %rax
100005d47: 48 89 84 24 88 00 00 00     	movq	%rax, 136(%rsp)
100005d4f: 4b 8d 04 c9                 	leaq	(%r9,%r9,8), %rax
100005d53: 48 8b 54 24 48              	movq	72(%rsp), %rdx
100005d58: 48 8d 0c 02                 	leaq	(%rdx,%rax), %rcx
100005d5c: 48 83 c1 09                 	addq	$9, %rcx
100005d60: 48 89 8c 24 80 00 00 00     	movq	%rcx, 128(%rsp)
100005d68: 48 8b 4c 24 50              	movq	80(%rsp), %rcx
100005d6d: 48 8d 5c 01 01              	leaq	1(%rcx,%rax), %rbx
100005d72: 48 89 5c 24 78              	movq	%rbx, 120(%rsp)
100005d77: 4c 8d 14 02                 	leaq	(%rdx,%rax), %r10
100005d7b: 4c 8d 1c 01                 	leaq	(%rcx,%rax), %r11
100005d7f: 48 8d 44 02 08              	leaq	8(%rdx,%rax), %rax
100005d84: 48 89 44 24 70              	movq	%rax, 112(%rsp)
100005d89: c4 c1 f9 6e c1              	vmovq	%r9, %xmm0
100005d8e: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005d93: 41 be 00 00 00 00           	movl	$0, %r14d
100005d99: 48 8b 44 24 38              	movq	56(%rsp), %rax
100005d9e: 48 89 44 24 30              	movq	%rax, 48(%rsp)
100005da3: 48 8b 44 24 40              	movq	64(%rsp), %rax
100005da8: 31 c9                       	xorl	%ecx, %ecx
100005daa: 31 d2                       	xorl	%edx, %edx
100005dac: 48 89 54 24 08              	movq	%rdx, 8(%rsp)
100005db1: 4c 89 4c 24 68              	movq	%r9, 104(%rsp)
100005db6: 48 89 7c 24 60              	movq	%rdi, 96(%rsp)
100005dbb: 4c 89 54 24 20              	movq	%r10, 32(%rsp)
100005dc0: 4c 89 5c 24 18              	movq	%r11, 24(%rsp)
100005dc5: c5 fd 7f 84 24 80 02 00 00  	vmovdqa	%ymm0, 640(%rsp)
100005dce: eb 38                       	jmp	56 <__ZN11LineNetwork7forwardEv+0x6f8>
100005dd0: 48 8b 8c 24 90 00 00 00     	movq	144(%rsp), %rcx
100005dd8: 48 ff c1                    	incq	%rcx
100005ddb: 48 8b 44 24 28              	movq	40(%rsp), %rax
100005de0: 48 05 00 04 00 00           	addq	$1024, %rax
100005de6: 48 81 44 24 30 00 04 00 00  	addq	$1024, 48(%rsp)
100005def: 49 81 c6 00 01 00 00        	addq	$256, %r14
100005df6: 48 81 7c 24 08 fd 01 00 00  	cmpq	$509, 8(%rsp)
100005dff: 4d 89 e9                    	movq	%r13, %r9
100005e02: 0f 83 28 ff ff ff           	jae	-216 <__ZN11LineNetwork7forwardEv+0x620>
100005e08: 48 89 44 24 28              	movq	%rax, 40(%rsp)
100005e0d: 4c 89 b4 24 98 00 00 00     	movq	%r14, 152(%rsp)
100005e15: 48 89 cb                    	movq	%rcx, %rbx
100005e18: 48 c1 e3 0b                 	shlq	$11, %rbx
100005e1c: 4d 89 cd                    	movq	%r9, %r13
100005e1f: 49 8d 04 19                 	leaq	(%r9,%rbx), %rax
100005e23: 4c 01 f8                    	addq	%r15, %rax
100005e26: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100005e2e: 4c 01 fb                    	addq	%r15, %rbx
100005e31: 48 89 ca                    	movq	%rcx, %rdx
100005e34: 48 c1 e2 0a                 	shlq	$10, %rdx
100005e38: 4c 8d 0c 16                 	leaq	(%rsi,%rdx), %r9
100005e3c: 49 81 c1 ff 05 00 00        	addq	$1535, %r9
100005e43: 48 01 f2                    	addq	%rsi, %rdx
100005e46: 4c 39 c8                    	cmpq	%r9, %rax
100005e49: 41 0f 92 c4                 	setb	%r12b
100005e4d: 48 39 da                    	cmpq	%rbx, %rdx
100005e50: 41 0f 92 c2                 	setb	%r10b
100005e54: 48 3b 84 24 80 00 00 00     	cmpq	128(%rsp), %rax
100005e5c: 41 0f 92 c6                 	setb	%r14b
100005e60: 48 39 5c 24 70              	cmpq	%rbx, 112(%rsp)
100005e65: 4c 89 da                    	movq	%r11, %rdx
100005e68: 41 0f 92 c3                 	setb	%r11b
100005e6c: 48 3b 44 24 78              	cmpq	120(%rsp), %rax
100005e71: 0f 92 c0                    	setb	%al
100005e74: 48 39 da                    	cmpq	%rbx, %rdx
100005e77: 41 0f 92 c1                 	setb	%r9b
100005e7b: 45 84 d4                    	testb	%r10b, %r12b
100005e7e: 48 89 8c 24 90 00 00 00     	movq	%rcx, 144(%rsp)
100005e86: 0f 85 84 0a 00 00           	jne	2692 <__ZN11LineNetwork7forwardEv+0x1200>
100005e8c: 45 20 de                    	andb	%r11b, %r14b
100005e8f: 0f 85 7b 0a 00 00           	jne	2683 <__ZN11LineNetwork7forwardEv+0x1200>
100005e95: ba 00 00 00 00              	movl	$0, %edx
100005e9a: 44 20 c8                    	andb	%r9b, %al
100005e9d: 0f 85 6f 0a 00 00           	jne	2671 <__ZN11LineNetwork7forwardEv+0x1202>
100005ea3: 48 8b 44 24 08              	movq	8(%rsp), %rax
100005ea8: 48 c1 e0 07                 	shlq	$7, %rax
100005eac: c4 e1 f9 6e c0              	vmovq	%rax, %xmm0
100005eb1: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005eb6: c5 fd 7f 84 24 a0 02 00 00  	vmovdqa	%ymm0, 672(%rsp)
100005ebf: 45 31 db                    	xorl	%r11d, %r11d
100005ec2: c5 fc 28 05 f6 13 00 00     	vmovaps	5110(%rip), %ymm0
100005eca: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100005ed3: c5 fc 28 05 c5 13 00 00     	vmovaps	5061(%rip), %ymm0
100005edb: c5 fc 29 84 24 20 02 00 00  	vmovaps	%ymm0, 544(%rsp)
100005ee4: c5 fc 28 05 94 13 00 00     	vmovaps	5012(%rip), %ymm0
100005eec: c5 fc 29 84 24 00 02 00 00  	vmovaps	%ymm0, 512(%rsp)
100005ef5: c5 fc 28 05 63 13 00 00     	vmovaps	4963(%rip), %ymm0
100005efd: c5 fc 29 84 24 e0 01 00 00  	vmovaps	%ymm0, 480(%rsp)
100005f06: c5 fc 28 05 32 13 00 00     	vmovaps	4914(%rip), %ymm0
100005f0e: c5 fc 29 84 24 c0 01 00 00  	vmovaps	%ymm0, 448(%rsp)
100005f17: c5 fc 28 05 01 13 00 00     	vmovaps	4865(%rip), %ymm0
100005f1f: c5 fc 29 84 24 a0 01 00 00  	vmovaps	%ymm0, 416(%rsp)
100005f28: c5 fc 28 05 d0 12 00 00     	vmovaps	4816(%rip), %ymm0
100005f30: c5 fc 29 84 24 80 01 00 00  	vmovaps	%ymm0, 384(%rsp)
100005f39: c5 fc 28 05 9f 12 00 00     	vmovaps	4767(%rip), %ymm0
100005f41: c5 fc 29 84 24 60 01 00 00  	vmovaps	%ymm0, 352(%rsp)
100005f4a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100005f50: 48 8b 4c 24 28              	movq	40(%rsp), %rcx
100005f55: c4 a1 7e 6f 84 59 1f fc ff ff       	vmovdqu	-993(%rcx,%r11,2), %ymm0
100005f5f: c4 e2 7d 00 c2              	vpshufb	%ymm2, %ymm0, %ymm0
100005f64: c4 a1 7e 6f 8c 59 ff fb ff ff       	vmovdqu	-1025(%rcx,%r11,2), %ymm1
100005f6e: c4 21 7e 6f 84 59 00 fc ff ff       	vmovdqu	-1024(%rcx,%r11,2), %ymm8
100005f78: c5 7d 6f 1d 80 13 00 00     	vmovdqa	4992(%rip), %ymm11
100005f80: c4 c2 75 00 cb              	vpshufb	%ymm11, %ymm1, %ymm1
100005f85: c4 e3 75 02 c0 cc           	vpblendd	$204, %ymm0, %ymm1, %ymm0
100005f8b: c4 e3 fd 00 c8 d8           	vpermq	$216, %ymm0, %ymm1
100005f91: c4 a1 7a 6f 94 59 0f fc ff ff       	vmovdqu	-1009(%rcx,%r11,2), %xmm2
100005f9b: c5 f9 6f 1d 0d 12 00 00     	vmovdqa	4621(%rip), %xmm3
100005fa3: c4 e2 69 00 d3              	vpshufb	%xmm3, %xmm2, %xmm2
100005fa8: c5 79 6f e3                 	vmovdqa	%xmm3, %xmm12
100005fac: c4 62 7d 21 ca              	vpmovsxbd	%xmm2, %ymm9
100005fb1: c4 63 fd 00 d0 db           	vpermq	$219, %ymm0, %ymm10
100005fb7: 48 8b 44 24 20              	movq	32(%rsp), %rax
100005fbc: c4 e2 79 78 00              	vpbroadcastb	(%rax), %xmm0
100005fc1: c4 e2 7d 21 d0              	vpmovsxbd	%xmm0, %ymm2
100005fc6: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
100005fcb: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100005fd4: c4 62 7d 21 c9              	vpmovsxbd	%xmm1, %ymm9
100005fd9: c4 42 7d 21 d2              	vpmovsxbd	%xmm10, %ymm10
100005fde: c4 21 7e 6f ac 59 20 fc ff ff       	vmovdqu	-992(%rcx,%r11,2), %ymm13
100005fe8: c4 62 15 00 3d ef 12 00 00  	vpshufb	4847(%rip), %ymm13, %ymm15
100005ff1: c4 c2 3d 00 fb              	vpshufb	%ymm11, %ymm8, %ymm7
100005ff6: c4 c3 45 02 ff cc           	vpblendd	$204, %ymm15, %ymm7, %ymm7
100005ffc: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
100006002: c4 63 fd 00 ff d8           	vpermq	$216, %ymm7, %ymm15
100006008: c5 fd 6f 05 10 13 00 00     	vmovdqa	4880(%rip), %ymm0
100006010: c4 62 15 00 e8              	vpshufb	%ymm0, %ymm13, %ymm13
100006015: c5 fd 6f 05 23 13 00 00     	vmovdqa	4899(%rip), %ymm0
10000601d: c4 62 3d 00 c0              	vpshufb	%ymm0, %ymm8, %ymm8
100006022: c4 c3 3d 02 f5 cc           	vpblendd	$204, %ymm13, %ymm8, %ymm6
100006028: c4 e3 fd 00 ee d8           	vpermq	$216, %ymm6, %ymm5
10000602e: c4 e2 7d 21 e1              	vpmovsxbd	%xmm1, %ymm4
100006033: c4 c2 7d 21 df              	vpmovsxbd	%xmm15, %ymm3
100006038: c4 e3 fd 00 cf db           	vpermq	$219, %ymm7, %ymm1
10000603e: c4 62 7d 21 e9              	vpmovsxbd	%xmm1, %ymm13
100006043: c4 43 7d 39 ff 01           	vextracti128	$1, %ymm15, %xmm15
100006049: c4 42 6d 40 c2              	vpmulld	%ymm10, %ymm2, %ymm8
10000604e: c4 21 7a 6f 94 59 10 fc ff ff       	vmovdqu	-1008(%rcx,%r11,2), %xmm10
100006058: c4 c2 29 00 fc              	vpshufb	%xmm12, %xmm10, %xmm7
10000605d: c4 62 79 78 70 01           	vpbroadcastb	1(%rax), %xmm14
100006063: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
100006068: c5 fd 7f 84 24 a0 00 00 00  	vmovdqa	%ymm0, 160(%rsp)
100006071: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
100006076: c4 42 7d 21 f6              	vpmovsxbd	%xmm14, %ymm14
10000607b: c4 e2 0d 40 ff              	vpmulld	%ymm7, %ymm14, %ymm7
100006080: c4 42 0d 40 ed              	vpmulld	%ymm13, %ymm14, %ymm13
100006085: c4 42 7d 21 e7              	vpmovsxbd	%xmm15, %ymm12
10000608a: c4 c3 7d 39 ef 01           	vextracti128	$1, %ymm5, %xmm15
100006090: c4 e3 fd 00 f6 db           	vpermq	$219, %ymm6, %ymm6
100006096: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
10000609b: c4 62 0d 40 cb              	vpmulld	%ymm3, %ymm14, %ymm9
1000060a0: c4 e2 7d 21 dd              	vpmovsxbd	%xmm5, %ymm3
1000060a5: c5 f9 6f 05 13 11 00 00     	vmovdqa	4371(%rip), %xmm0
1000060ad: c4 e2 29 00 e8              	vpshufb	%xmm0, %xmm10, %xmm5
1000060b2: c4 e2 79 78 40 02           	vpbroadcastb	2(%rax), %xmm0
1000060b8: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
1000060bd: c4 c2 7d 21 cf              	vpmovsxbd	%xmm15, %ymm1
1000060c2: c4 62 7d 40 fb              	vpmulld	%ymm3, %ymm0, %ymm15
1000060c7: c4 62 7d 40 d6              	vpmulld	%ymm6, %ymm0, %ymm10
1000060cc: c4 e2 6d 40 d4              	vpmulld	%ymm4, %ymm2, %ymm2
1000060d1: c5 fd 7f 94 24 40 01 00 00  	vmovdqa	%ymm2, 320(%rsp)
1000060da: c4 e2 7d 21 d5              	vpmovsxbd	%xmm5, %ymm2
1000060df: c4 e2 7d 40 d2              	vpmulld	%ymm2, %ymm0, %ymm2
1000060e4: c4 a1 7e 6f 9c 59 ff fd ff ff       	vmovdqu	-513(%rcx,%r11,2), %ymm3
1000060ee: c4 c2 0d 40 e4              	vpmulld	%ymm12, %ymm14, %ymm4
1000060f3: c5 fd 7f a4 24 00 01 00 00  	vmovdqa	%ymm4, 256(%rsp)
1000060fc: c4 a1 7e 6f a4 59 1f fe ff ff       	vmovdqu	-481(%rcx,%r11,2), %ymm4
100006106: c4 e2 5d 00 25 d1 11 00 00  	vpshufb	4561(%rip), %ymm4, %ymm4
10000610f: c4 c2 65 00 db              	vpshufb	%ymm11, %ymm3, %ymm3
100006114: c4 e3 65 02 dc cc           	vpblendd	$204, %ymm4, %ymm3, %ymm3
10000611a: c4 e2 7d 40 c1              	vpmulld	%ymm1, %ymm0, %ymm0
10000611f: c5 fd 7f 84 24 20 01 00 00  	vmovdqa	%ymm0, 288(%rsp)
100006128: c4 e3 fd 00 c3 d8           	vpermq	$216, %ymm3, %ymm0
10000612e: c4 e2 7d 21 c8              	vpmovsxbd	%xmm0, %ymm1
100006133: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
100006139: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
10000613e: c4 e3 7d 39 c0 01           	vextracti128	$1, %ymm0, %xmm0
100006144: c5 c5 fe a4 24 c0 00 00 00  	vpaddd	192(%rsp), %ymm7, %ymm4
10000614d: c4 a1 7a 6f ac 59 0f fe ff ff       	vmovdqu	-497(%rcx,%r11,2), %xmm5
100006157: c5 79 6f 35 51 10 00 00     	vmovdqa	4177(%rip), %xmm14
10000615f: c4 c2 51 00 ee              	vpshufb	%xmm14, %xmm5, %xmm5
100006164: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006169: c4 e2 79 78 70 03           	vpbroadcastb	3(%rax), %xmm6
10000616f: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006174: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
100006179: c4 e2 4d 40 c0              	vpmulld	%ymm0, %ymm6, %ymm0
10000617e: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100006187: c4 e2 4d 40 db              	vpmulld	%ymm3, %ymm6, %ymm3
10000618c: c4 41 3d fe ed              	vpaddd	%ymm13, %ymm8, %ymm13
100006191: c4 e2 4d 40 c9              	vpmulld	%ymm1, %ymm6, %ymm1
100006196: c4 e2 4d 40 c5              	vpmulld	%ymm5, %ymm6, %ymm0
10000619b: c5 ed fe c0                 	vpaddd	%ymm0, %ymm2, %ymm0
10000619f: c5 35 fe 84 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm9, %ymm8
1000061a8: c5 dd fe c0                 	vpaddd	%ymm0, %ymm4, %ymm0
1000061ac: c5 fd 7f 84 24 e0 00 00 00  	vmovdqa	%ymm0, 224(%rsp)
1000061b5: c4 a1 7e 6f a4 59 00 fe ff ff       	vmovdqu	-512(%rcx,%r11,2), %ymm4
1000061bf: c4 a1 7e 6f ac 59 20 fe ff ff       	vmovdqu	-480(%rcx,%r11,2), %ymm5
1000061c9: c4 e2 55 00 35 0e 11 00 00  	vpshufb	4366(%rip), %ymm5, %ymm6
1000061d2: c4 c2 5d 00 fb              	vpshufb	%ymm11, %ymm4, %ymm7
1000061d7: c5 2d fe d3                 	vpaddd	%ymm3, %ymm10, %ymm10
1000061db: c4 e3 45 02 de cc           	vpblendd	$204, %ymm6, %ymm7, %ymm3
1000061e1: c4 e3 fd 00 f3 d8           	vpermq	$216, %ymm3, %ymm6
1000061e7: c4 e3 7d 39 f7 01           	vextracti128	$1, %ymm6, %xmm7
1000061ed: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
1000061f2: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
1000061f8: c5 05 fe c9                 	vpaddd	%ymm1, %ymm15, %ymm9
1000061fc: c4 e2 7d 21 cb              	vpmovsxbd	%xmm3, %ymm1
100006201: c4 e2 7d 21 de              	vpmovsxbd	%xmm6, %ymm3
100006206: c4 a1 7a 6f b4 59 10 fe ff ff       	vmovdqu	-496(%rcx,%r11,2), %xmm6
100006210: c4 e2 79 78 40 04           	vpbroadcastb	4(%rax), %xmm0
100006216: c4 c2 49 00 d6              	vpshufb	%xmm14, %xmm6, %xmm2
10000621b: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
100006220: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006225: c4 e2 7d 40 db              	vpmulld	%ymm3, %ymm0, %ymm3
10000622a: c4 62 7d 40 e1              	vpmulld	%ymm1, %ymm0, %ymm12
10000622f: c4 e2 7d 40 cf              	vpmulld	%ymm7, %ymm0, %ymm1
100006234: c5 fd 7f 8c 24 a0 00 00 00  	vmovdqa	%ymm1, 160(%rsp)
10000623d: c4 e2 7d 40 c2              	vpmulld	%ymm2, %ymm0, %ymm0
100006242: c4 e2 55 00 15 d5 10 00 00  	vpshufb	4309(%rip), %ymm5, %ymm2
10000624b: c4 e2 5d 00 25 ec 10 00 00  	vpshufb	4332(%rip), %ymm4, %ymm4
100006254: c4 e3 5d 02 d2 cc           	vpblendd	$204, %ymm2, %ymm4, %ymm2
10000625a: c4 e2 79 78 60 05           	vpbroadcastb	5(%rax), %xmm4
100006260: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100006265: c4 e3 fd 00 ea db           	vpermq	$219, %ymm2, %ymm5
10000626b: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006270: c4 e2 5d 40 ed              	vpmulld	%ymm5, %ymm4, %ymm5
100006275: c5 9d fe ed                 	vpaddd	%ymm5, %ymm12, %ymm5
100006279: c4 e3 fd 00 d2 d8           	vpermq	$216, %ymm2, %ymm2
10000627f: c4 e2 7d 21 fa              	vpmovsxbd	%xmm2, %ymm7
100006284: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000628a: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000628f: c4 e2 49 00 35 28 0f 00 00  	vpshufb	3880(%rip), %xmm6, %xmm6
100006298: c4 e2 5d 40 ff              	vpmulld	%ymm7, %ymm4, %ymm7
10000629d: c4 62 5d 40 fa              	vpmulld	%ymm2, %ymm4, %ymm15
1000062a2: c4 e2 7d 21 d6              	vpmovsxbd	%xmm6, %ymm2
1000062a7: c4 e2 5d 40 d2              	vpmulld	%ymm2, %ymm4, %ymm2
1000062ac: c5 fd fe c2                 	vpaddd	%ymm2, %ymm0, %ymm0
1000062b0: c4 a1 7e 6f 54 59 ff        	vmovdqu	-1(%rcx,%r11,2), %ymm2
1000062b7: c5 e5 fe df                 	vpaddd	%ymm7, %ymm3, %ymm3
1000062bb: c4 a1 7e 6f 64 59 1f        	vmovdqu	31(%rcx,%r11,2), %ymm4
1000062c2: c4 e2 5d 00 25 15 10 00 00  	vpshufb	4117(%rip), %ymm4, %ymm4
1000062cb: c4 c2 6d 00 d3              	vpshufb	%ymm11, %ymm2, %ymm2
1000062d0: c4 e3 6d 02 d4 cc           	vpblendd	$204, %ymm4, %ymm2, %ymm2
1000062d6: c4 e3 fd 00 e2 d8           	vpermq	$216, %ymm2, %ymm4
1000062dc: c4 c1 15 fe f2              	vpaddd	%ymm10, %ymm13, %ymm6
1000062e1: c4 e3 7d 39 e7 01           	vextracti128	$1, %ymm4, %xmm7
1000062e7: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
1000062ec: c4 e3 fd 00 d2 db           	vpermq	$219, %ymm2, %ymm2
1000062f2: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000062f7: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000062fc: c4 41 3d fe c1              	vpaddd	%ymm9, %ymm8, %ymm8
100006301: c4 e2 79 78 48 06           	vpbroadcastb	6(%rax), %xmm1
100006307: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
10000630c: c4 e2 75 40 e4              	vpmulld	%ymm4, %ymm1, %ymm4
100006311: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
100006315: c4 a1 7a 6f 64 59 0f        	vmovdqu	15(%rcx,%r11,2), %xmm4
10000631c: c4 c2 59 00 e6              	vpshufb	%xmm14, %xmm4, %xmm4
100006321: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100006326: c4 e2 75 40 d2              	vpmulld	%ymm2, %ymm1, %ymm2
10000632b: c5 d5 fe d2                 	vpaddd	%ymm2, %ymm5, %ymm2
10000632f: c4 62 75 40 ef              	vpmulld	%ymm7, %ymm1, %ymm13
100006334: c4 e2 75 40 cc              	vpmulld	%ymm4, %ymm1, %ymm1
100006339: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
10000633d: c5 3d fe c3                 	vpaddd	%ymm3, %ymm8, %ymm8
100006341: c5 7d fe 94 24 e0 00 00 00  	vpaddd	224(%rsp), %ymm0, %ymm10
10000634a: c4 a1 7e 6f 0c 59           	vmovdqu	(%rcx,%r11,2), %ymm1
100006350: c4 a1 7e 6f 5c 59 20        	vmovdqu	32(%rcx,%r11,2), %ymm3
100006357: c4 e2 65 00 25 80 0f 00 00  	vpshufb	3968(%rip), %ymm3, %ymm4
100006360: c4 c2 75 00 eb              	vpshufb	%ymm11, %ymm1, %ymm5
100006365: c5 4d fe da                 	vpaddd	%ymm2, %ymm6, %ymm11
100006369: c4 e3 55 02 e4 cc           	vpblendd	$204, %ymm4, %ymm5, %ymm4
10000636f: c4 e3 fd 00 ec d8           	vpermq	$216, %ymm4, %ymm5
100006375: c4 e2 65 00 1d a2 0f 00 00  	vpshufb	4002(%rip), %ymm3, %ymm3
10000637e: c4 e2 75 00 0d b9 0f 00 00  	vpshufb	4025(%rip), %ymm1, %ymm1
100006387: c4 e3 75 02 cb cc           	vpblendd	$204, %ymm3, %ymm1, %ymm1
10000638d: c5 fd 6f 84 24 00 01 00 00  	vmovdqa	256(%rsp), %ymm0
100006396: c5 7d fe a4 24 40 01 00 00  	vpaddd	320(%rsp), %ymm0, %ymm12
10000639f: c4 e3 fd 00 f1 d8           	vpermq	$216, %ymm1, %ymm6
1000063a5: c4 e2 7d 21 fd              	vpmovsxbd	%xmm5, %ymm7
1000063aa: c4 e3 fd 00 e4 db           	vpermq	$219, %ymm4, %ymm4
1000063b0: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000063b5: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
1000063bb: c5 fd 6f 84 24 c0 00 00 00  	vmovdqa	192(%rsp), %ymm0
1000063c4: c5 7d fe 8c 24 20 01 00 00  	vpaddd	288(%rsp), %ymm0, %ymm9
1000063cd: c4 a1 7a 6f 44 59 10        	vmovdqu	16(%rcx,%r11,2), %xmm0
1000063d4: c4 c2 79 00 d6              	vpshufb	%xmm14, %xmm0, %xmm2
1000063d9: c4 e2 79 78 58 07           	vpbroadcastb	7(%rax), %xmm3
1000063df: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000063e4: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
1000063e9: c4 e2 65 40 e4              	vpmulld	%ymm4, %ymm3, %ymm4
1000063ee: c4 e2 65 40 ff              	vpmulld	%ymm7, %ymm3, %ymm7
1000063f3: c4 e2 65 40 ed              	vpmulld	%ymm5, %ymm3, %ymm5
1000063f8: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000063fd: c4 e2 65 40 d2              	vpmulld	%ymm2, %ymm3, %ymm2
100006402: c4 e2 79 78 58 08           	vpbroadcastb	8(%rax), %xmm3
100006408: c5 05 fe b4 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm15, %ymm14
100006411: c4 62 7d 21 fe              	vpmovsxbd	%xmm6, %ymm15
100006416: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
10000641b: c4 42 65 40 ff              	vpmulld	%ymm15, %ymm3, %ymm15
100006420: c4 c1 45 fe ff              	vpaddd	%ymm15, %ymm7, %ymm7
100006425: c4 41 1d fe c9              	vpaddd	%ymm9, %ymm12, %ymm9
10000642a: c4 e3 fd 00 c9 db           	vpermq	$219, %ymm1, %ymm1
100006430: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
100006435: c4 e2 65 40 c9              	vpmulld	%ymm1, %ymm3, %ymm1
10000643a: c5 dd fe c9                 	vpaddd	%ymm1, %ymm4, %ymm1
10000643e: c4 c1 0d fe e5              	vpaddd	%ymm13, %ymm14, %ymm4
100006443: c4 e3 7d 39 f6 01           	vextracti128	$1, %ymm6, %xmm6
100006449: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
10000644e: c4 e2 65 40 f6              	vpmulld	%ymm6, %ymm3, %ymm6
100006453: c5 d5 fe ee                 	vpaddd	%ymm6, %ymm5, %ymm5
100006457: c4 e2 79 00 05 60 0d 00 00  	vpshufb	3424(%rip), %xmm0, %xmm0
100006460: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006465: c4 e2 65 40 c0              	vpmulld	%ymm0, %ymm3, %ymm0
10000646a: c5 ed fe c0                 	vpaddd	%ymm0, %ymm2, %ymm0
10000646e: 48 8b 44 24 18              	movq	24(%rsp), %rax
100006473: c4 e2 79 78 10              	vpbroadcastb	(%rax), %xmm2
100006478: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000647d: c5 c5 fe da                 	vpaddd	%ymm2, %ymm7, %ymm3
100006481: c5 bd fe db                 	vpaddd	%ymm3, %ymm8, %ymm3
100006485: c5 f5 fe ca                 	vpaddd	%ymm2, %ymm1, %ymm1
100006489: c5 a5 fe c9                 	vpaddd	%ymm1, %ymm11, %ymm1
10000648d: c5 b5 fe e4                 	vpaddd	%ymm4, %ymm9, %ymm4
100006491: c5 d5 fe ea                 	vpaddd	%ymm2, %ymm5, %ymm5
100006495: c5 fd fe c2                 	vpaddd	%ymm2, %ymm0, %ymm0
100006499: c5 ad fe c0                 	vpaddd	%ymm0, %ymm10, %ymm0
10000649d: c5 fd 6f b4 24 60 02 00 00  	vmovdqa	608(%rsp), %ymm6
1000064a6: c4 e2 75 40 ce              	vpmulld	%ymm6, %ymm1, %ymm1
1000064ab: c5 dd fe d5                 	vpaddd	%ymm5, %ymm4, %ymm2
1000064af: c4 e2 65 40 de              	vpmulld	%ymm6, %ymm3, %ymm3
1000064b4: c4 e2 7d 40 c6              	vpmulld	%ymm6, %ymm0, %ymm0
1000064b9: c4 e2 6d 40 d6              	vpmulld	%ymm6, %ymm2, %ymm2
1000064be: c5 dd 72 e3 1f              	vpsrad	$31, %ymm3, %ymm4
1000064c3: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
1000064c8: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
1000064cc: c5 dd 72 e1 1f              	vpsrad	$31, %ymm1, %ymm4
1000064d1: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
1000064d6: c5 e5 72 e3 0e              	vpsrad	$14, %ymm3, %ymm3
1000064db: c5 f5 fe cc                 	vpaddd	%ymm4, %ymm1, %ymm1
1000064df: c5 f5 72 e1 0e              	vpsrad	$14, %ymm1, %ymm1
1000064e4: c5 dd 72 e2 1f              	vpsrad	$31, %ymm2, %ymm4
1000064e9: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
1000064ee: c5 ed fe d4                 	vpaddd	%ymm4, %ymm2, %ymm2
1000064f2: c5 ed 72 e2 0e              	vpsrad	$14, %ymm2, %ymm2
1000064f7: c5 dd 72 e0 1f              	vpsrad	$31, %ymm0, %ymm4
1000064fc: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
100006501: c5 fd fe c4                 	vpaddd	%ymm4, %ymm0, %ymm0
100006505: c5 fd 72 e0 0e              	vpsrad	$14, %ymm0, %ymm0
10000650a: c4 e2 7d 58 25 cd 27 00 00  	vpbroadcastd	10189(%rip), %ymm4
100006513: c4 e2 6d 39 d4              	vpminsd	%ymm4, %ymm2, %ymm2
100006518: c4 e2 75 39 cc              	vpminsd	%ymm4, %ymm1, %ymm1
10000651d: c4 e2 65 39 dc              	vpminsd	%ymm4, %ymm3, %ymm3
100006522: c4 e2 7d 39 e4              	vpminsd	%ymm4, %ymm0, %ymm4
100006527: c4 e2 7d 58 2d b4 27 00 00  	vpbroadcastd	10164(%rip), %ymm5
100006530: c4 e2 75 3d c5              	vpmaxsd	%ymm5, %ymm1, %ymm0
100006535: c4 e2 6d 3d cd              	vpmaxsd	%ymm5, %ymm2, %ymm1
10000653a: c5 f5 6b c0                 	vpackssdw	%ymm0, %ymm1, %ymm0
10000653e: c4 e2 65 3d cd              	vpmaxsd	%ymm5, %ymm3, %ymm1
100006543: c4 e2 5d 3d d5              	vpmaxsd	%ymm5, %ymm4, %ymm2
100006548: c5 f5 6b ca                 	vpackssdw	%ymm2, %ymm1, %ymm1
10000654c: c5 fd 6f b4 24 40 02 00 00  	vmovdqa	576(%rsp), %ymm6
100006555: c5 ed 73 d6 01              	vpsrlq	$1, %ymm6, %ymm2
10000655a: c5 fd 6f ac 24 a0 02 00 00  	vmovdqa	672(%rsp), %ymm5
100006563: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006567: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000656c: c5 fd 6f a4 24 80 02 00 00  	vmovdqa	640(%rsp), %ymm4
100006575: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006579: c4 c1 f9 7e d2              	vmovq	%xmm2, %r10
10000657e: c4 e3 f9 16 d0 01           	vpextrq	$1, %xmm2, %rax
100006584: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000658a: c4 c1 f9 7e d0              	vmovq	%xmm2, %r8
10000658f: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
100006595: c5 fd 6f bc 24 20 02 00 00  	vmovdqa	544(%rsp), %ymm7
10000659e: c5 ed 73 d7 01              	vpsrlq	$1, %ymm7, %ymm2
1000065a3: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
1000065a7: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000065ac: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000065b0: c4 c1 f9 7e d5              	vmovq	%xmm2, %r13
1000065b5: c4 c3 f9 16 d6 01           	vpextrq	$1, %xmm2, %r14
1000065bb: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000065c1: c4 e1 f9 7e d6              	vmovq	%xmm2, %rsi
1000065c6: c4 e3 f9 16 d7 01           	vpextrq	$1, %xmm2, %rdi
1000065cc: c5 7d 6f 84 24 00 02 00 00  	vmovdqa	512(%rsp), %ymm8
1000065d5: c4 c1 6d 73 d0 01           	vpsrlq	$1, %ymm8, %ymm2
1000065db: c4 e3 fd 00 c0 d8           	vpermq	$216, %ymm0, %ymm0
1000065e1: c4 e3 fd 00 c9 d8           	vpermq	$216, %ymm1, %ymm1
1000065e7: c5 f5 63 c0                 	vpacksswb	%ymm0, %ymm1, %ymm0
1000065eb: c5 7d 6f 8c 24 e0 01 00 00  	vmovdqa	480(%rsp), %ymm9
1000065f4: c4 c1 65 73 d1 01           	vpsrlq	$1, %ymm9, %ymm3
1000065fa: c5 ed d4 cd                 	vpaddq	%ymm5, %ymm2, %ymm1
1000065fe: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
100006603: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100006607: c4 e3 f9 16 8c 24 40 01 00 00 01    	vpextrq	$1, %xmm1, 320(%rsp)
100006612: c4 e1 f9 7e ca              	vmovq	%xmm1, %rdx
100006617: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
10000661d: c5 f9 d6 8c 24 20 01 00 00  	vmovq	%xmm1, 288(%rsp)
100006626: c4 c3 f9 16 cc 01           	vpextrq	$1, %xmm1, %r12
10000662c: c5 7d 6f 94 24 c0 01 00 00  	vmovdqa	448(%rsp), %ymm10
100006635: c4 c1 75 73 d2 01           	vpsrlq	$1, %ymm10, %ymm1
10000663b: c5 e5 d4 d5                 	vpaddq	%ymm5, %ymm3, %ymm2
10000663f: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006644: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006648: c4 83 79 14 04 17 00        	vpextrb	$0, %xmm0, (%r15,%r10)
10000664f: c4 e1 f9 7e d1              	vmovq	%xmm2, %rcx
100006654: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
10000665a: c4 c3 79 14 04 07 01        	vpextrb	$1, %xmm0, (%r15,%rax)
100006661: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006667: c4 e3 f9 16 94 24 00 01 00 00 01    	vpextrq	$1, %xmm2, 256(%rsp)
100006672: c4 83 79 14 04 07 02        	vpextrb	$2, %xmm0, (%r15,%r8)
100006679: c4 e1 f9 7e d3              	vmovq	%xmm2, %rbx
10000667e: c5 fd 6f 9c 24 a0 01 00 00  	vmovdqa	416(%rsp), %ymm3
100006687: c5 ed 73 d3 01              	vpsrlq	$1, %ymm3, %ymm2
10000668c: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006690: c5 f5 d4 cd                 	vpaddq	%ymm5, %ymm1, %ymm1
100006694: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
100006699: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000669e: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000066a2: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000066a6: c4 83 79 14 04 0f 03        	vpextrb	$3, %xmm0, (%r15,%r9)
1000066ad: c4 c1 f9 7e c8              	vmovq	%xmm1, %r8
1000066b2: c4 83 79 14 04 2f 04        	vpextrb	$4, %xmm0, (%r15,%r13)
1000066b9: c4 e3 f9 16 8c 24 c0 00 00 00 01    	vpextrq	$1, %xmm1, 192(%rsp)
1000066c4: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
1000066ca: c4 83 79 14 04 37 05        	vpextrb	$5, %xmm0, (%r15,%r14)
1000066d1: c4 c3 79 14 04 37 06        	vpextrb	$6, %xmm0, (%r15,%rsi)
1000066d8: c4 c1 f9 7e ce              	vmovq	%xmm1, %r14
1000066dd: c4 e3 f9 16 8c 24 e0 00 00 00 01    	vpextrq	$1, %xmm1, 224(%rsp)
1000066e8: c4 c3 79 14 04 3f 07        	vpextrb	$7, %xmm0, (%r15,%rdi)
1000066ef: c4 e3 f9 16 94 24 a0 00 00 00 01    	vpextrq	$1, %xmm2, 160(%rsp)
1000066fa: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006700: c4 c3 79 14 0c 17 00        	vpextrb	$0, %xmm1, (%r15,%rdx)
100006707: c4 e1 f9 7e d7              	vmovq	%xmm2, %rdi
10000670c: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006712: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
10000671a: c4 c3 79 14 0c 07 01        	vpextrb	$1, %xmm1, (%r15,%rax)
100006721: c4 c1 f9 7e d5              	vmovq	%xmm2, %r13
100006726: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
10000672e: c4 c3 79 14 0c 07 02        	vpextrb	$2, %xmm1, (%r15,%rax)
100006735: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
10000673b: c5 7d 6f 9c 24 80 01 00 00  	vmovdqa	384(%rsp), %ymm11
100006744: c4 c1 6d 73 d3 01           	vpsrlq	$1, %ymm11, %ymm2
10000674a: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
10000674e: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006753: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006757: c4 83 79 14 0c 27 03        	vpextrb	$3, %xmm1, (%r15,%r12)
10000675e: c4 e1 f9 7e d2              	vmovq	%xmm2, %rdx
100006763: c4 c3 79 14 0c 0f 04        	vpextrb	$4, %xmm1, (%r15,%rcx)
10000676a: c4 e3 f9 16 d1 01           	vpextrq	$1, %xmm2, %rcx
100006770: c4 83 79 14 0c 17 05        	vpextrb	$5, %xmm1, (%r15,%r10)
100006777: c4 c3 79 14 0c 1f 06        	vpextrb	$6, %xmm1, (%r15,%rbx)
10000677e: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006784: c4 e1 f9 7e d3              	vmovq	%xmm2, %rbx
100006789: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
10000678f: c5 7d 6f a4 24 60 01 00 00  	vmovdqa	352(%rsp), %ymm12
100006798: c4 c1 6d 73 d4 01           	vpsrlq	$1, %ymm12, %ymm2
10000679e: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
1000067a2: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000067a7: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000067ab: 48 8b 84 24 00 01 00 00     	movq	256(%rsp), %rax
1000067b3: c4 c3 79 14 0c 07 07        	vpextrb	$7, %xmm1, (%r15,%rax)
1000067ba: c4 e1 f9 7e d0              	vmovq	%xmm2, %rax
1000067bf: c4 83 79 14 04 07 08        	vpextrb	$8, %xmm0, (%r15,%r8)
1000067c6: c4 c3 f9 16 d0 01           	vpextrq	$1, %xmm2, %r8
1000067cc: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000067d2: 48 8b b4 24 c0 00 00 00     	movq	192(%rsp), %rsi
1000067da: c4 c3 79 14 04 37 09        	vpextrb	$9, %xmm0, (%r15,%rsi)
1000067e1: c4 83 79 14 04 37 0a        	vpextrb	$10, %xmm0, (%r15,%r14)
1000067e8: c4 c1 f9 7e d6              	vmovq	%xmm2, %r14
1000067ed: c4 c3 f9 16 d4 01           	vpextrq	$1, %xmm2, %r12
1000067f3: c5 fd 6f 15 e5 0a 00 00     	vmovdqa	2789(%rip), %ymm2
1000067fb: 48 8b b4 24 e0 00 00 00     	movq	224(%rsp), %rsi
100006803: c4 c3 79 14 04 37 0b        	vpextrb	$11, %xmm0, (%r15,%rsi)
10000680a: c4 c3 79 14 04 3f 0c        	vpextrb	$12, %xmm0, (%r15,%rdi)
100006811: 48 8b b4 24 a0 00 00 00     	movq	160(%rsp), %rsi
100006819: c4 c3 79 14 04 37 0d        	vpextrb	$13, %xmm0, (%r15,%rsi)
100006820: c4 83 79 14 04 2f 0e        	vpextrb	$14, %xmm0, (%r15,%r13)
100006827: c4 83 79 14 04 0f 0f        	vpextrb	$15, %xmm0, (%r15,%r9)
10000682e: c4 c3 79 14 0c 17 08        	vpextrb	$8, %xmm1, (%r15,%rdx)
100006835: c4 c3 79 14 0c 0f 09        	vpextrb	$9, %xmm1, (%r15,%rcx)
10000683c: c4 c3 79 14 0c 1f 0a        	vpextrb	$10, %xmm1, (%r15,%rbx)
100006843: c4 83 79 14 0c 17 0b        	vpextrb	$11, %xmm1, (%r15,%r10)
10000684a: c4 c3 79 14 0c 07 0c        	vpextrb	$12, %xmm1, (%r15,%rax)
100006851: c4 83 79 14 0c 07 0d        	vpextrb	$13, %xmm1, (%r15,%r8)
100006858: c4 83 79 14 0c 37 0e        	vpextrb	$14, %xmm1, (%r15,%r14)
10000685f: c4 83 79 14 0c 27 0f        	vpextrb	$15, %xmm1, (%r15,%r12)
100006866: c4 e2 7d 59 05 79 24 00 00  	vpbroadcastq	9337(%rip), %ymm0
10000686f: c5 cd d4 f0                 	vpaddq	%ymm0, %ymm6, %ymm6
100006873: c5 fd 7f b4 24 40 02 00 00  	vmovdqa	%ymm6, 576(%rsp)
10000687c: c5 c5 d4 f8                 	vpaddq	%ymm0, %ymm7, %ymm7
100006880: c5 fd 7f bc 24 20 02 00 00  	vmovdqa	%ymm7, 544(%rsp)
100006889: c5 3d d4 c0                 	vpaddq	%ymm0, %ymm8, %ymm8
10000688d: c5 7d 7f 84 24 00 02 00 00  	vmovdqa	%ymm8, 512(%rsp)
100006896: c5 35 d4 c8                 	vpaddq	%ymm0, %ymm9, %ymm9
10000689a: c5 7d 7f 8c 24 e0 01 00 00  	vmovdqa	%ymm9, 480(%rsp)
1000068a3: c5 2d d4 d0                 	vpaddq	%ymm0, %ymm10, %ymm10
1000068a7: c5 7d 7f 94 24 c0 01 00 00  	vmovdqa	%ymm10, 448(%rsp)
1000068b0: c5 e5 d4 d8                 	vpaddq	%ymm0, %ymm3, %ymm3
1000068b4: c5 fd 7f 9c 24 a0 01 00 00  	vmovdqa	%ymm3, 416(%rsp)
1000068bd: c5 25 d4 d8                 	vpaddq	%ymm0, %ymm11, %ymm11
1000068c1: c5 7d 7f 9c 24 80 01 00 00  	vmovdqa	%ymm11, 384(%rsp)
1000068ca: c5 1d d4 e0                 	vpaddq	%ymm0, %ymm12, %ymm12
1000068ce: c5 7d 7f a4 24 60 01 00 00  	vmovdqa	%ymm12, 352(%rsp)
1000068d7: 49 83 c3 20                 	addq	$32, %r11
1000068db: 49 81 fb e0 00 00 00        	cmpq	$224, %r11
1000068e2: 0f 85 68 f6 ff ff           	jne	-2456 <__ZN11LineNetwork7forwardEv+0x840>
1000068e8: ba c0 01 00 00              	movl	$448, %edx
1000068ed: 44 8b 44 24 14              	movl	20(%rsp), %r8d
1000068f2: 48 8b 74 24 58              	movq	88(%rsp), %rsi
1000068f7: 4c 8b 6c 24 68              	movq	104(%rsp), %r13
1000068fc: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100006901: eb 0f                       	jmp	15 <__ZN11LineNetwork7forwardEv+0x1202>
100006903: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000690d: 0f 1f 00                    	nopl	(%rax)
100006910: 31 d2                       	xorl	%edx, %edx
100006912: 48 83 44 24 08 02           	addq	$2, 8(%rsp)
100006918: 48 89 d0                    	movq	%rdx, %rax
10000691b: 48 d1 e8                    	shrq	%rax
10000691e: 4c 8b b4 24 98 00 00 00     	movq	152(%rsp), %r14
100006926: 4c 01 f0                    	addq	%r14, %rax
100006929: 4c 8d 0c c7                 	leaq	(%rdi,%rax,8), %r9
10000692d: 4c 8b 54 24 20              	movq	32(%rsp), %r10
100006932: 4c 8b 5c 24 18              	movq	24(%rsp), %r11
100006937: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x1248>
100006939: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100006940: 41 88 09                    	movb	%cl, (%r9)
100006943: 48 83 c2 02                 	addq	$2, %rdx
100006947: 49 83 c1 08                 	addq	$8, %r9
10000694b: 48 81 fa fd 01 00 00        	cmpq	$509, %rdx
100006952: 0f 83 78 f4 ff ff           	jae	-2952 <__ZN11LineNetwork7forwardEv+0x6c0>
100006958: 4c 8b 64 24 30              	movq	48(%rsp), %r12
10000695d: 41 0f be 8c 14 fe fb ff ff  	movsbl	-1026(%r12,%rdx), %ecx
100006966: 41 0f be 02                 	movsbl	(%r10), %eax
10000696a: 0f af c1                    	imull	%ecx, %eax
10000696d: 41 0f be 8c 14 ff fb ff ff  	movsbl	-1025(%r12,%rdx), %ecx
100006976: 41 0f be 5a 01              	movsbl	1(%r10), %ebx
10000697b: 0f af d9                    	imull	%ecx, %ebx
10000697e: 01 c3                       	addl	%eax, %ebx
100006980: 41 0f be 8c 14 00 fc ff ff  	movsbl	-1024(%r12,%rdx), %ecx
100006989: 41 0f be 42 02              	movsbl	2(%r10), %eax
10000698e: 0f af c1                    	imull	%ecx, %eax
100006991: 01 d8                       	addl	%ebx, %eax
100006993: 41 0f be 8c 14 fe fd ff ff  	movsbl	-514(%r12,%rdx), %ecx
10000699c: 41 0f be 5a 03              	movsbl	3(%r10), %ebx
1000069a1: 0f af d9                    	imull	%ecx, %ebx
1000069a4: 01 c3                       	addl	%eax, %ebx
1000069a6: 41 0f be 8c 14 ff fd ff ff  	movsbl	-513(%r12,%rdx), %ecx
1000069af: 41 0f be 42 04              	movsbl	4(%r10), %eax
1000069b4: 0f af c1                    	imull	%ecx, %eax
1000069b7: 01 d8                       	addl	%ebx, %eax
1000069b9: 41 0f be 8c 14 00 fe ff ff  	movsbl	-512(%r12,%rdx), %ecx
1000069c2: 41 0f be 5a 05              	movsbl	5(%r10), %ebx
1000069c7: 0f af d9                    	imull	%ecx, %ebx
1000069ca: 01 c3                       	addl	%eax, %ebx
1000069cc: 41 0f be 4c 14 fe           	movsbl	-2(%r12,%rdx), %ecx
1000069d2: 41 0f be 42 06              	movsbl	6(%r10), %eax
1000069d7: 0f af c1                    	imull	%ecx, %eax
1000069da: 01 d8                       	addl	%ebx, %eax
1000069dc: 41 0f be 4c 14 ff           	movsbl	-1(%r12,%rdx), %ecx
1000069e2: 41 0f be 5a 07              	movsbl	7(%r10), %ebx
1000069e7: 0f af d9                    	imull	%ecx, %ebx
1000069ea: 01 c3                       	addl	%eax, %ebx
1000069ec: 41 0f be 0c 14              	movsbl	(%r12,%rdx), %ecx
1000069f1: 41 0f be 42 08              	movsbl	8(%r10), %eax
1000069f6: 0f af c1                    	imull	%ecx, %eax
1000069f9: 01 d8                       	addl	%ebx, %eax
1000069fb: 41 0f be 1b                 	movsbl	(%r11), %ebx
1000069ff: 01 c3                       	addl	%eax, %ebx
100006a01: 41 0f af d8                 	imull	%r8d, %ebx
100006a05: 89 d9                       	movl	%ebx, %ecx
100006a07: c1 f9 1f                    	sarl	$31, %ecx
100006a0a: c1 e9 12                    	shrl	$18, %ecx
100006a0d: 01 d9                       	addl	%ebx, %ecx
100006a0f: c1 f9 0e                    	sarl	$14, %ecx
100006a12: 81 f9 80 00 00 00           	cmpl	$128, %ecx
100006a18: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x130f>
100006a1a: b9 7f 00 00 00              	movl	$127, %ecx
100006a1f: 83 f9 81                    	cmpl	$-127, %ecx
100006a22: 0f 8f 18 ff ff ff           	jg	-232 <__ZN11LineNetwork7forwardEv+0x1230>
100006a28: b9 81 00 00 00              	movl	$129, %ecx
100006a2d: e9 0e ff ff ff              	jmp	-242 <__ZN11LineNetwork7forwardEv+0x1230>
100006a32: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100006a36: 5b                          	popq	%rbx
100006a37: 41 5c                       	popq	%r12
100006a39: 41 5d                       	popq	%r13
100006a3b: 41 5e                       	popq	%r14
100006a3d: 41 5f                       	popq	%r15
100006a3f: 5d                          	popq	%rbp
100006a40: c5 f8 77                    	vzeroupper
100006a43: c3                          	retq
100006a44: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100006a4e: 66 90                       	nop
100006a50: 55                          	pushq	%rbp
100006a51: 48 89 e5                    	movq	%rsp, %rbp
100006a54: 5d                          	popq	%rbp
100006a55: e9 16 df ff ff              	jmp	-8426 <__ZN14ModelInterfaceD2Ev>
100006a5a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100006a60: 55                          	pushq	%rbp
100006a61: 48 89 e5                    	movq	%rsp, %rbp
100006a64: 53                          	pushq	%rbx
100006a65: 50                          	pushq	%rax
100006a66: 48 89 fb                    	movq	%rdi, %rbx
100006a69: e8 02 df ff ff              	callq	-8446 <__ZN14ModelInterfaceD2Ev>
100006a6e: 48 89 df                    	movq	%rbx, %rdi
100006a71: 48 83 c4 08                 	addq	$8, %rsp
100006a75: 5b                          	popq	%rbx
100006a76: 5d                          	popq	%rbp
100006a77: e9 16 04 00 00              	jmp	1046 <dyld_stub_binder+0x100006e92>
100006a7c: 0f 1f 40 00                 	nopl	(%rax)
100006a80: 55                          	pushq	%rbp
100006a81: 48 89 e5                    	movq	%rsp, %rbp
100006a84: 48 83 e4 e0                 	andq	$-32, %rsp
100006a88: 48 81 ec a0 00 00 00        	subq	$160, %rsp
100006a8f: 48 8b 05 c2 25 00 00        	movq	9666(%rip), %rax
100006a96: 48 8b 00                    	movq	(%rax), %rax
100006a99: 48 89 84 24 88 00 00 00     	movq	%rax, 136(%rsp)
100006aa1: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100006aa5: c5 fc 29 44 24 60           	vmovaps	%ymm0, 96(%rsp)
100006aab: c5 fc 29 44 24 40           	vmovaps	%ymm0, 64(%rsp)
100006ab1: c5 fc 29 44 24 20           	vmovaps	%ymm0, 32(%rsp)
100006ab7: c5 fc 29 04 24              	vmovaps	%ymm0, (%rsp)
100006abc: c4 e2 7d 21 17              	vpmovsxbd	(%rdi), %ymm2
100006ac1: c4 e2 7d 21 1e              	vpmovsxbd	(%rsi), %ymm3
100006ac6: c4 e2 7d 21 47 08           	vpmovsxbd	8(%rdi), %ymm0
100006acc: c4 e2 7d 21 4e 08           	vpmovsxbd	8(%rsi), %ymm1
100006ad2: c4 e2 75 40 c8              	vpmulld	%ymm0, %ymm1, %ymm1
100006ad7: c5 fd 7f 4c 24 20           	vmovdqa	%ymm1, 32(%rsp)
100006add: c4 e2 7d 21 67 10           	vpmovsxbd	16(%rdi), %ymm4
100006ae3: c4 e2 7d 21 6e 10           	vpmovsxbd	16(%rsi), %ymm5
100006ae9: c5 fd 6f 44 24 40           	vmovdqa	64(%rsp), %ymm0
100006aef: c5 fd fe 44 24 60           	vpaddd	96(%rsp), %ymm0, %ymm0
100006af5: 48 8b 05 5c 25 00 00        	movq	9564(%rip), %rax
100006afc: 48 8b 00                    	movq	(%rax), %rax
100006aff: 48 3b 84 24 88 00 00 00     	cmpq	136(%rsp), %rax
100006b07: 0f 85 ba 00 00 00           	jne	186 <__ZN11LineNetwork7forwardEv+0x14b7>
100006b0d: c4 e2 65 40 d2              	vpmulld	%ymm2, %ymm3, %ymm2
100006b12: c4 e2 55 40 dc              	vpmulld	%ymm4, %ymm5, %ymm3
100006b17: c5 e5 fe d2                 	vpaddd	%ymm2, %ymm3, %ymm2
100006b1b: c5 f9 7e d0                 	vmovd	%xmm2, %eax
100006b1f: c4 e3 79 16 d1 01           	vpextrd	$1, %xmm2, %ecx
100006b25: c4 e3 79 16 d2 02           	vpextrd	$2, %xmm2, %edx
100006b2b: 01 c1                       	addl	%eax, %ecx
100006b2d: 01 ca                       	addl	%ecx, %edx
100006b2f: c4 e3 79 16 d0 03           	vpextrd	$3, %xmm2, %eax
100006b35: 01 d0                       	addl	%edx, %eax
100006b37: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006b3d: c5 f9 7e d1                 	vmovd	%xmm2, %ecx
100006b41: 01 c1                       	addl	%eax, %ecx
100006b43: c4 e3 79 16 d0 01           	vpextrd	$1, %xmm2, %eax
100006b49: c4 e3 79 16 d2 02           	vpextrd	$2, %xmm2, %edx
100006b4f: 01 c8                       	addl	%ecx, %eax
100006b51: 01 c2                       	addl	%eax, %edx
100006b53: c4 e3 79 16 d0 03           	vpextrd	$3, %xmm2, %eax
100006b59: 01 d0                       	addl	%edx, %eax
100006b5b: c5 f9 7e c9                 	vmovd	%xmm1, %ecx
100006b5f: 01 c1                       	addl	%eax, %ecx
100006b61: c4 e3 79 16 c8 01           	vpextrd	$1, %xmm1, %eax
100006b67: 01 c8                       	addl	%ecx, %eax
100006b69: c4 e3 79 16 c9 02           	vpextrd	$2, %xmm1, %ecx
100006b6f: 01 c1                       	addl	%eax, %ecx
100006b71: c4 e3 79 16 c8 03           	vpextrd	$3, %xmm1, %eax
100006b77: 01 c8                       	addl	%ecx, %eax
100006b79: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
100006b7f: c5 f9 7e c9                 	vmovd	%xmm1, %ecx
100006b83: 01 c1                       	addl	%eax, %ecx
100006b85: c4 e3 79 16 c8 01           	vpextrd	$1, %xmm1, %eax
100006b8b: 01 c8                       	addl	%ecx, %eax
100006b8d: c4 e3 79 16 c9 02           	vpextrd	$2, %xmm1, %ecx
100006b93: 01 c1                       	addl	%eax, %ecx
100006b95: c4 e3 79 16 ca 03           	vpextrd	$3, %xmm1, %edx
100006b9b: 01 ca                       	addl	%ecx, %edx
100006b9d: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006ba3: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006ba7: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006bac: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006bb0: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006bb5: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006bb9: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006bbd: 01 d0                       	addl	%edx, %eax
100006bbf: 48 89 ec                    	movq	%rbp, %rsp
100006bc2: 5d                          	popq	%rbp
100006bc3: c5 f8 77                    	vzeroupper
100006bc6: c3                          	retq
100006bc7: c5 f8 77                    	vzeroupper
100006bca: e8 e1 02 00 00              	callq	737 <dyld_stub_binder+0x100006eb0>
100006bcf: 90                          	nop
100006bd0: 55                          	pushq	%rbp
100006bd1: 48 89 e5                    	movq	%rsp, %rbp
100006bd4: 48 83 e4 e0                 	andq	$-32, %rsp
100006bd8: 48 81 ec a0 00 00 00        	subq	$160, %rsp
100006bdf: 48 8b 05 72 24 00 00        	movq	9330(%rip), %rax
100006be6: 48 8b 00                    	movq	(%rax), %rax
100006be9: 48 89 84 24 88 00 00 00     	movq	%rax, 136(%rsp)
100006bf1: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100006bf5: c5 fc 29 44 24 60           	vmovaps	%ymm0, 96(%rsp)
100006bfb: c5 fc 29 44 24 40           	vmovaps	%ymm0, 64(%rsp)
100006c01: c5 fc 29 44 24 20           	vmovaps	%ymm0, 32(%rsp)
100006c07: c5 fc 29 04 24              	vmovaps	%ymm0, (%rsp)
100006c0c: c4 e2 7d 21 07              	vpmovsxbd	(%rdi), %ymm0
100006c11: c4 e2 7d 21 0e              	vpmovsxbd	(%rsi), %ymm1
100006c16: c4 e2 75 40 d0              	vpmulld	%ymm0, %ymm1, %ymm2
100006c1b: c4 e2 7d 21 47 08           	vpmovsxbd	8(%rdi), %ymm0
100006c21: c4 e2 7d 21 4e 08           	vpmovsxbd	8(%rsi), %ymm1
100006c27: c4 e2 75 40 d8              	vpmulld	%ymm0, %ymm1, %ymm3
100006c2c: c4 e2 7d 21 47 10           	vpmovsxbd	16(%rdi), %ymm0
100006c32: c4 e2 7d 21 4e 10           	vpmovsxbd	16(%rsi), %ymm1
100006c38: c4 e2 75 40 c0              	vpmulld	%ymm0, %ymm1, %ymm0
100006c3d: c4 e2 7d 21 4f 18           	vpmovsxbd	24(%rdi), %ymm1
100006c43: c4 e2 7d 21 66 18           	vpmovsxbd	24(%rsi), %ymm4
100006c49: c4 e2 5d 40 c9              	vpmulld	%ymm1, %ymm4, %ymm1
100006c4e: c5 fd 7f 14 24              	vmovdqa	%ymm2, (%rsp)
100006c53: c5 fd 7f 5c 24 20           	vmovdqa	%ymm3, 32(%rsp)
100006c59: c5 fd 7f 44 24 40           	vmovdqa	%ymm0, 64(%rsp)
100006c5f: c5 fd 7f 4c 24 60           	vmovdqa	%ymm1, 96(%rsp)
100006c65: c4 e2 7d 21 67 20           	vpmovsxbd	32(%rdi), %ymm4
100006c6b: c4 e2 7d 21 76 20           	vpmovsxbd	32(%rsi), %ymm6
100006c71: c4 e2 7d 21 6f 28           	vpmovsxbd	40(%rdi), %ymm5
100006c77: c4 e2 7d 21 7e 28           	vpmovsxbd	40(%rsi), %ymm7
100006c7d: 48 8b 05 d4 23 00 00        	movq	9172(%rip), %rax
100006c84: 48 8b 00                    	movq	(%rax), %rax
100006c87: 48 3b 84 24 88 00 00 00     	cmpq	136(%rsp), %rax
100006c8f: 0f 85 c2 00 00 00           	jne	194 <__ZN11LineNetwork7forwardEv+0x1647>
100006c95: c4 e2 4d 40 e4              	vpmulld	%ymm4, %ymm6, %ymm4
100006c9a: c5 dd fe e2                 	vpaddd	%ymm2, %ymm4, %ymm4
100006c9e: c4 e2 45 40 d5              	vpmulld	%ymm5, %ymm7, %ymm2
100006ca3: c5 ed fe d3                 	vpaddd	%ymm3, %ymm2, %ymm2
100006ca7: c5 f9 7e e0                 	vmovd	%xmm4, %eax
100006cab: c4 e3 79 16 e1 01           	vpextrd	$1, %xmm4, %ecx
100006cb1: 01 c1                       	addl	%eax, %ecx
100006cb3: c4 e3 79 16 e0 02           	vpextrd	$2, %xmm4, %eax
100006cb9: 01 c8                       	addl	%ecx, %eax
100006cbb: c4 e3 79 16 e1 03           	vpextrd	$3, %xmm4, %ecx
100006cc1: 01 c1                       	addl	%eax, %ecx
100006cc3: c4 e3 7d 39 e3 01           	vextracti128	$1, %ymm4, %xmm3
100006cc9: c5 f9 7e d8                 	vmovd	%xmm3, %eax
100006ccd: 01 c8                       	addl	%ecx, %eax
100006ccf: c4 e3 79 16 d9 01           	vpextrd	$1, %xmm3, %ecx
100006cd5: 01 c1                       	addl	%eax, %ecx
100006cd7: c4 e3 79 16 d8 02           	vpextrd	$2, %xmm3, %eax
100006cdd: 01 c8                       	addl	%ecx, %eax
100006cdf: c4 e3 79 16 d9 03           	vpextrd	$3, %xmm3, %ecx
100006ce5: 01 c1                       	addl	%eax, %ecx
100006ce7: c5 f9 7e d0                 	vmovd	%xmm2, %eax
100006ceb: 01 c8                       	addl	%ecx, %eax
100006ced: c4 e3 79 16 d1 01           	vpextrd	$1, %xmm2, %ecx
100006cf3: 01 c1                       	addl	%eax, %ecx
100006cf5: c4 e3 79 16 d0 02           	vpextrd	$2, %xmm2, %eax
100006cfb: 01 c8                       	addl	%ecx, %eax
100006cfd: c4 e3 79 16 d1 03           	vpextrd	$3, %xmm2, %ecx
100006d03: 01 c1                       	addl	%eax, %ecx
100006d05: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006d0b: c5 f9 7e d0                 	vmovd	%xmm2, %eax
100006d0f: 01 c8                       	addl	%ecx, %eax
100006d11: c4 e3 79 16 d1 01           	vpextrd	$1, %xmm2, %ecx
100006d17: 01 c1                       	addl	%eax, %ecx
100006d19: c4 e3 79 16 d0 02           	vpextrd	$2, %xmm2, %eax
100006d1f: 01 c8                       	addl	%ecx, %eax
100006d21: c4 e3 79 16 d1 03           	vpextrd	$3, %xmm2, %ecx
100006d27: 01 c1                       	addl	%eax, %ecx
100006d29: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006d2d: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006d33: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006d37: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100006d3c: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006d40: c5 f9 70 c8 e5              	vpshufd	$229, %xmm0, %xmm1
100006d45: c5 f9 fe c1                 	vpaddd	%xmm1, %xmm0, %xmm0
100006d49: c5 f9 7e c0                 	vmovd	%xmm0, %eax
100006d4d: 01 c8                       	addl	%ecx, %eax
100006d4f: 48 89 ec                    	movq	%rbp, %rsp
100006d52: 5d                          	popq	%rbp
100006d53: c5 f8 77                    	vzeroupper
100006d56: c3                          	retq
100006d57: c5 f8 77                    	vzeroupper
100006d5a: e8 51 01 00 00              	callq	337 <dyld_stub_binder+0x100006eb0>
100006d5f: 90                          	nop
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
