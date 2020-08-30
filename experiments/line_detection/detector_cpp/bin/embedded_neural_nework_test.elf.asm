
bin/embedded_neural_nework_test.elf:	file format Mach-O 64-bit x86-64


Disassembly of section __TEXT,__text:

0000000100002710 __Z8get_timev:
100002710: 55                          	pushq	%rbp
100002711: 48 89 e5                    	movq	%rsp, %rbp
100002714: e8 73 47 00 00              	callq	18291 <dyld_stub_binder+0x100006e8c>
100002719: c4 e1 fb 2a c0              	vcvtsi2sd	%rax, %xmm0, %xmm0
10000271e: c5 fb 5e 05 3a 49 00 00     	vdivsd	18746(%rip), %xmm0, %xmm0
100002726: 5d                          	popq	%rbp
100002727: c3                          	retq
100002728: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100002730 __Z14get_predictionRN2cv3MatER14ModelInterfacef:
100002730: 55                          	pushq	%rbp
100002731: 48 89 e5                    	movq	%rsp, %rbp
100002734: 41 57                       	pushq	%r15
100002736: 41 56                       	pushq	%r14
100002738: 41 55                       	pushq	%r13
10000273a: 41 54                       	pushq	%r12
10000273c: 53                          	pushq	%rbx
10000273d: 48 81 ec 28 01 00 00        	subq	$296, %rsp
100002744: c5 fa 11 45 a8              	vmovss	%xmm0, -88(%rbp)
100002749: 49 89 d6                    	movq	%rdx, %r14
10000274c: 49 89 f4                    	movq	%rsi, %r12
10000274f: 48 89 fb                    	movq	%rdi, %rbx
100002752: 48 8b 05 07 69 00 00        	movq	26887(%rip), %rax
100002759: 48 8b 00                    	movq	(%rax), %rax
10000275c: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100002760: 8b 46 08                    	movl	8(%rsi), %eax
100002763: 8b 4e 0c                    	movl	12(%rsi), %ecx
100002766: c7 85 d0 fe ff ff 00 00 ff 42       	movl	$1124007936, -304(%rbp)
100002770: 48 8d 95 d8 fe ff ff        	leaq	-296(%rbp), %rdx
100002777: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000277b: c5 fc 11 85 d4 fe ff ff     	vmovups	%ymm0, -300(%rbp)
100002783: c5 fc 11 85 f0 fe ff ff     	vmovups	%ymm0, -272(%rbp)
10000278b: 48 89 95 10 ff ff ff        	movq	%rdx, -240(%rbp)
100002792: 48 8d 95 20 ff ff ff        	leaq	-224(%rbp), %rdx
100002799: 48 89 95 18 ff ff ff        	movq	%rdx, -232(%rbp)
1000027a0: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000027a4: c5 f8 11 85 20 ff ff ff     	vmovups	%xmm0, -224(%rbp)
1000027ac: 89 4d b8                    	movl	%ecx, -72(%rbp)
1000027af: 89 45 bc                    	movl	%eax, -68(%rbp)
1000027b2: 4c 8d bd d0 fe ff ff        	leaq	-304(%rbp), %r15
1000027b9: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
1000027bd: 4c 89 ff                    	movq	%r15, %rdi
1000027c0: be 02 00 00 00              	movl	$2, %esi
1000027c5: 31 c9                       	xorl	%ecx, %ecx
1000027c7: c5 f8 77                    	vzeroupper
1000027ca: e8 51 46 00 00              	callq	18001 <dyld_stub_binder+0x100006e20>
1000027cf: 48 c7 85 40 ff ff ff 00 00 00 00    	movq	$0, -192(%rbp)
1000027da: c7 85 30 ff ff ff 00 00 01 01       	movl	$16842752, -208(%rbp)
1000027e4: 4c 89 a5 38 ff ff ff        	movq	%r12, -200(%rbp)
1000027eb: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
1000027f3: c7 45 b8 00 00 01 02        	movl	$33619968, -72(%rbp)
1000027fa: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
1000027fe: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002805: 48 8d 75 b8                 	leaq	-72(%rbp), %rsi
100002809: ba 06 00 00 00              	movl	$6, %edx
10000280e: 31 c9                       	xorl	%ecx, %ecx
100002810: e8 35 46 00 00              	callq	17973 <dyld_stub_binder+0x100006e4a>
100002815: 41 8b 44 24 08              	movl	8(%r12), %eax
10000281a: 41 8b 4c 24 0c              	movl	12(%r12), %ecx
10000281f: c7 85 30 ff ff ff 00 00 ff 42       	movl	$1124007936, -208(%rbp)
100002829: 48 8d 95 38 ff ff ff        	leaq	-200(%rbp), %rdx
100002830: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002834: c5 fc 11 85 34 ff ff ff     	vmovups	%ymm0, -204(%rbp)
10000283c: c5 fc 11 85 50 ff ff ff     	vmovups	%ymm0, -176(%rbp)
100002844: 48 89 95 70 ff ff ff        	movq	%rdx, -144(%rbp)
10000284b: 48 8d 55 80                 	leaq	-128(%rbp), %rdx
10000284f: 48 89 95 78 ff ff ff        	movq	%rdx, -136(%rbp)
100002856: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000285a: c5 f8 11 45 80              	vmovups	%xmm0, -128(%rbp)
10000285f: 89 4d b8                    	movl	%ecx, -72(%rbp)
100002862: 89 45 bc                    	movl	%eax, -68(%rbp)
100002865: 4c 8d a5 30 ff ff ff        	leaq	-208(%rbp), %r12
10000286c: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
100002870: 4c 89 e7                    	movq	%r12, %rdi
100002873: be 02 00 00 00              	movl	$2, %esi
100002878: 31 c9                       	xorl	%ecx, %ecx
10000287a: c5 f8 77                    	vzeroupper
10000287d: e8 9e 45 00 00              	callq	17822 <dyld_stub_binder+0x100006e20>
100002882: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
10000288a: c7 45 b8 00 00 01 01        	movl	$16842752, -72(%rbp)
100002891: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
100002895: 48 c7 85 c0 fe ff ff 00 00 00 00    	movq	$0, -320(%rbp)
1000028a0: c7 85 b0 fe ff ff 00 00 01 02       	movl	$33619968, -336(%rbp)
1000028aa: 4c 89 a5 b8 fe ff ff        	movq	%r12, -328(%rbp)
1000028b1: 41 8b 46 0c                 	movl	12(%r14), %eax
1000028b5: 41 8b 4e 10                 	movl	16(%r14), %ecx
1000028b9: 89 4d 90                    	movl	%ecx, -112(%rbp)
1000028bc: 89 45 94                    	movl	%eax, -108(%rbp)
1000028bf: 48 8d 7d b8                 	leaq	-72(%rbp), %rdi
1000028c3: 48 8d b5 b0 fe ff ff        	leaq	-336(%rbp), %rsi
1000028ca: 48 8d 55 90                 	leaq	-112(%rbp), %rdx
1000028ce: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000028d2: c5 f0 57 c9                 	vxorps	%xmm1, %xmm1, %xmm1
1000028d6: b9 03 00 00 00              	movl	$3, %ecx
1000028db: e8 58 45 00 00              	callq	17752 <dyld_stub_binder+0x100006e38>
1000028e0: 41 8b 46 0c                 	movl	12(%r14), %eax
1000028e4: 85 c0                       	testl	%eax, %eax
1000028e6: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
1000028ea: 4d 89 f7                    	movq	%r14, %r15
1000028ed: 0f 84 7c 00 00 00           	je	124 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x23f>
1000028f3: 41 8b 4f 10                 	movl	16(%r15), %ecx
1000028f7: 31 d2                       	xorl	%edx, %edx
1000028f9: 45 31 e4                    	xorl	%r12d, %r12d
1000028fc: 85 c9                       	testl	%ecx, %ecx
1000028fe: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1dc>
100002900: 31 c9                       	xorl	%ecx, %ecx
100002902: ff c2                       	incl	%edx
100002904: 39 c2                       	cmpl	%eax, %edx
100002906: 73 67                       	jae	103 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x23f>
100002908: 85 c9                       	testl	%ecx, %ecx
10000290a: 74 f4                       	je	-12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1d0>
10000290c: 89 55 a0                    	movl	%edx, -96(%rbp)
10000290f: 4c 63 f2                    	movslq	%edx, %r14
100002912: 45 31 ed                    	xorl	%r13d, %r13d
100002915: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000291f: 90                          	nop
100002920: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
100002927: 48 8b 00                    	movq	(%rax), %rax
10000292a: 49 0f af c6                 	imulq	%r14, %rax
10000292e: 48 03 85 40 ff ff ff        	addq	-192(%rbp), %rax
100002935: 49 63 cd                    	movslq	%r13d, %rcx
100002938: 0f b6 1c 01                 	movzbl	(%rcx,%rax), %ebx
10000293c: 4c 89 ff                    	movq	%r15, %rdi
10000293f: e8 4c 21 00 00              	callq	8524 <__ZN14ModelInterface12input_bufferEv>
100002944: 43 8d 0c 2c                 	leal	(%r12,%r13), %ecx
100002948: d0 eb                       	shrb	%bl
10000294a: 89 c9                       	movl	%ecx, %ecx
10000294c: 88 1c 08                    	movb	%bl, (%rax,%rcx)
10000294f: 41 ff c5                    	incl	%r13d
100002952: 41 8b 4f 10                 	movl	16(%r15), %ecx
100002956: 41 39 cd                    	cmpl	%ecx, %r13d
100002959: 72 c5                       	jb	-59 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1f0>
10000295b: 41 8b 47 0c                 	movl	12(%r15), %eax
10000295f: 45 01 ec                    	addl	%r13d, %r12d
100002962: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002966: 8b 55 a0                    	movl	-96(%rbp), %edx
100002969: ff c2                       	incl	%edx
10000296b: 39 c2                       	cmpl	%eax, %edx
10000296d: 72 99                       	jb	-103 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x1d8>
10000296f: 49 8b 07                    	movq	(%r15), %rax
100002972: 4c 89 ff                    	movq	%r15, %rdi
100002975: ff 50 10                    	callq	*16(%rax)
100002978: 41 8b 47 18                 	movl	24(%r15), %eax
10000297c: 41 8b 4f 1c                 	movl	28(%r15), %ecx
100002980: c7 03 00 00 ff 42           	movl	$1124007936, (%rbx)
100002986: 48 8d 53 08                 	leaq	8(%rbx), %rdx
10000298a: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000298e: c5 fc 11 43 04              	vmovups	%ymm0, 4(%rbx)
100002993: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
100002998: 48 89 53 40                 	movq	%rdx, 64(%rbx)
10000299c: 48 8d 53 50                 	leaq	80(%rbx), %rdx
1000029a0: 48 89 95 c8 fe ff ff        	movq	%rdx, -312(%rbp)
1000029a7: 48 89 53 48                 	movq	%rdx, 72(%rbx)
1000029ab: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000029af: c5 f8 11 43 50              	vmovups	%xmm0, 80(%rbx)
1000029b4: 89 4d b8                    	movl	%ecx, -72(%rbp)
1000029b7: 89 45 bc                    	movl	%eax, -68(%rbp)
1000029ba: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
1000029be: 48 89 df                    	movq	%rbx, %rdi
1000029c1: be 02 00 00 00              	movl	$2, %esi
1000029c6: 31 c9                       	xorl	%ecx, %ecx
1000029c8: c5 f8 77                    	vzeroupper
1000029cb: e8 50 44 00 00              	callq	17488 <dyld_stub_binder+0x100006e20>
1000029d0: 41 8b 47 18                 	movl	24(%r15), %eax
1000029d4: 41 83 7f 14 01              	cmpl	$1, 20(%r15)
1000029d9: 4d 89 fc                    	movq	%r15, %r12
1000029dc: 0f 85 c7 00 00 00           	jne	199 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x379>
1000029e2: 85 c0                       	testl	%eax, %eax
1000029e4: 0f 84 e2 01 00 00           	je	482 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
1000029ea: c5 fa 10 45 a8              	vmovss	-88(%rbp), %xmm0
1000029ef: c5 fa 59 05 b1 46 00 00     	vmulss	18097(%rip), %xmm0, %xmm0
1000029f7: c5 fa 11 45 a0              	vmovss	%xmm0, -96(%rbp)
1000029fc: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002a01: 45 31 ff                    	xorl	%r15d, %r15d
100002a04: 31 d2                       	xorl	%edx, %edx
100002a06: 45 31 ed                    	xorl	%r13d, %r13d
100002a09: 85 c9                       	testl	%ecx, %ecx
100002a0b: 75 13                       	jne	19 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2f0>
100002a0d: 0f 1f 00                    	nopl	(%rax)
100002a10: 31 c9                       	xorl	%ecx, %ecx
100002a12: ff c2                       	incl	%edx
100002a14: 39 c2                       	cmpl	%eax, %edx
100002a16: 0f 83 b0 01 00 00           	jae	432 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002a1c: 85 c9                       	testl	%ecx, %ecx
100002a1e: 74 f0                       	je	-16 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2e0>
100002a20: 89 55 a8                    	movl	%edx, -88(%rbp)
100002a23: 4c 63 f2                    	movslq	%edx, %r14
100002a26: 31 db                       	xorl	%ebx, %ebx
100002a28: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002a30: 4c 89 e7                    	movq	%r12, %rdi
100002a33: e8 68 20 00 00              	callq	8296 <__ZN14ModelInterface13output_bufferEv>
100002a38: 42 8d 0c 2b                 	leal	(%rbx,%r13), %ecx
100002a3c: 89 c9                       	movl	%ecx, %ecx
100002a3e: 0f b6 04 08                 	movzbl	(%rax,%rcx), %eax
100002a42: 84 c0                       	testb	%al, %al
100002a44: 41 0f 48 c7                 	cmovsl	%r15d, %eax
100002a48: 0f be c8                    	movsbl	%al, %ecx
100002a4b: c5 ea 2a c1                 	vcvtsi2ss	%ecx, %xmm2, %xmm0
100002a4f: 48 8b 55 b0                 	movq	-80(%rbp), %rdx
100002a53: 48 8b 4a 48                 	movq	72(%rdx), %rcx
100002a57: 48 8b 09                    	movq	(%rcx), %rcx
100002a5a: 49 0f af ce                 	imulq	%r14, %rcx
100002a5e: 48 03 4a 10                 	addq	16(%rdx), %rcx
100002a62: 48 63 db                    	movslq	%ebx, %rbx
100002a65: 88 04 0b                    	movb	%al, (%rbx,%rcx)
100002a68: 48 8b 42 48                 	movq	72(%rdx), %rax
100002a6c: 48 8b 00                    	movq	(%rax), %rax
100002a6f: 49 0f af c6                 	imulq	%r14, %rax
100002a73: 48 03 42 10                 	addq	16(%rdx), %rax
100002a77: c5 f8 2e 45 a0              	vucomiss	-96(%rbp), %xmm0
100002a7c: 0f 97 04 03                 	seta	(%rbx,%rax)
100002a80: ff c3                       	incl	%ebx
100002a82: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002a87: 39 cb                       	cmpl	%ecx, %ebx
100002a89: 72 a5                       	jb	-91 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x300>
100002a8b: 41 8b 44 24 18              	movl	24(%r12), %eax
100002a90: 41 01 dd                    	addl	%ebx, %r13d
100002a93: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002a97: 8b 55 a8                    	movl	-88(%rbp), %edx
100002a9a: ff c2                       	incl	%edx
100002a9c: 39 c2                       	cmpl	%eax, %edx
100002a9e: 0f 82 78 ff ff ff           	jb	-136 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x2ec>
100002aa4: e9 23 01 00 00              	jmp	291 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002aa9: 85 c0                       	testl	%eax, %eax
100002aab: 0f 84 1b 01 00 00           	je	283 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002ab1: c5 fa 10 45 a8              	vmovss	-88(%rbp), %xmm0
100002ab6: c5 fa 59 05 ea 45 00 00     	vmulss	17898(%rip), %xmm0, %xmm0
100002abe: c5 fa 11 45 98              	vmovss	%xmm0, -104(%rbp)
100002ac3: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002ac8: 31 d2                       	xorl	%edx, %edx
100002aca: 45 31 ff                    	xorl	%r15d, %r15d
100002acd: 85 c9                       	testl	%ecx, %ecx
100002acf: 75 29                       	jne	41 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3ca>
100002ad1: e9 ea 00 00 00              	jmp	234 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x490>
100002ad6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002ae0: 41 8b 44 24 18              	movl	24(%r12), %eax
100002ae5: 8b 55 9c                    	movl	-100(%rbp), %edx
100002ae8: ff c2                       	incl	%edx
100002aea: 39 c2                       	cmpl	%eax, %edx
100002aec: 0f 83 da 00 00 00           	jae	218 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x49c>
100002af2: 85 c9                       	testl	%ecx, %ecx
100002af4: 0f 84 c6 00 00 00           	je	198 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x490>
100002afa: 89 55 9c                    	movl	%edx, -100(%rbp)
100002afd: 48 63 c2                    	movslq	%edx, %rax
100002b00: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100002b04: 31 d2                       	xorl	%edx, %edx
100002b06: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100002b0a: 41 83 7c 24 14 00           	cmpl	$0, 20(%r12)
100002b10: 75 60                       	jne	96 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x442>
100002b12: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002b1c: 0f 1f 40 00                 	nopl	(%rax)
100002b20: 41 b6 81                    	movb	$-127, %r14b
100002b23: 45 31 ed                    	xorl	%r13d, %r13d
100002b26: 41 0f be c6                 	movsbl	%r14b, %eax
100002b2a: c5 ea 2a c0                 	vcvtsi2ss	%eax, %xmm2, %xmm0
100002b2e: c5 f8 2e 45 98              	vucomiss	-104(%rbp), %xmm0
100002b33: b8 00 00 00 00              	movl	$0, %eax
100002b38: 44 0f 46 e8                 	cmovbel	%eax, %r13d
100002b3c: 48 8b 43 48                 	movq	72(%rbx), %rax
100002b40: 48 8b 00                    	movq	(%rax), %rax
100002b43: 48 0f af 45 a8              	imulq	-88(%rbp), %rax
100002b48: 48 03 43 10                 	addq	16(%rbx), %rax
100002b4c: 48 8b 55 a0                 	movq	-96(%rbp), %rdx
100002b50: 48 63 d2                    	movslq	%edx, %rdx
100002b53: 44 88 2c 02                 	movb	%r13b, (%rdx,%rax)
100002b57: ff c2                       	incl	%edx
100002b59: 41 8b 4c 24 1c              	movl	28(%r12), %ecx
100002b5e: 39 ca                       	cmpl	%ecx, %edx
100002b60: 0f 83 7a ff ff ff           	jae	-134 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3b0>
100002b66: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100002b6a: 41 83 7c 24 14 00           	cmpl	$0, 20(%r12)
100002b70: 74 ae                       	je	-82 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3f0>
100002b72: 41 b6 81                    	movb	$-127, %r14b
100002b75: 31 db                       	xorl	%ebx, %ebx
100002b77: 45 31 ed                    	xorl	%r13d, %r13d
100002b7a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100002b80: 4c 89 e7                    	movq	%r12, %rdi
100002b83: e8 18 1f 00 00              	callq	7960 <__ZN14ModelInterface13output_bufferEv>
100002b88: 41 8d 0c 1f                 	leal	(%r15,%rbx), %ecx
100002b8c: 89 c9                       	movl	%ecx, %ecx
100002b8e: 0f b6 04 08                 	movzbl	(%rax,%rcx), %eax
100002b92: 44 38 f0                    	cmpb	%r14b, %al
100002b95: 44 0f 4f eb                 	cmovgl	%ebx, %r13d
100002b99: 45 0f b6 f6                 	movzbl	%r14b, %r14d
100002b9d: 44 0f 4d f0                 	cmovgel	%eax, %r14d
100002ba1: ff c3                       	incl	%ebx
100002ba3: 41 3b 5c 24 14              	cmpl	20(%r12), %ebx
100002ba8: 72 d6                       	jb	-42 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x450>
100002baa: 41 01 df                    	addl	%ebx, %r15d
100002bad: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002bb1: e9 70 ff ff ff              	jmp	-144 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3f6>
100002bb6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002bc0: 31 c9                       	xorl	%ecx, %ecx
100002bc2: ff c2                       	incl	%edx
100002bc4: 39 c2                       	cmpl	%eax, %edx
100002bc6: 0f 82 26 ff ff ff           	jb	-218 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x3c2>
100002bcc: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002bd3: 48 85 c0                    	testq	%rax, %rax
100002bd6: 74 12                       	je	18 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4ba>
100002bd8: f0                          	lock
100002bd9: ff 48 14                    	decl	20(%rax)
100002bdc: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4ba>
100002bde: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002be5: e8 30 42 00 00              	callq	16944 <dyld_stub_binder+0x100006e1a>
100002bea: 48 c7 85 68 ff ff ff 00 00 00 00    	movq	$0, -152(%rbp)
100002bf5: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002bf9: c5 fc 11 85 40 ff ff ff     	vmovups	%ymm0, -192(%rbp)
100002c01: 83 bd 34 ff ff ff 00        	cmpl	$0, -204(%rbp)
100002c08: 7e 2c                       	jle	44 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x506>
100002c0a: 48 8b 85 70 ff ff ff        	movq	-144(%rbp), %rax
100002c11: 31 c9                       	xorl	%ecx, %ecx
100002c13: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002c1d: 0f 1f 00                    	nopl	(%rax)
100002c20: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002c27: 48 ff c1                    	incq	%rcx
100002c2a: 48 63 95 34 ff ff ff        	movslq	-204(%rbp), %rdx
100002c31: 48 39 d1                    	cmpq	%rdx, %rcx
100002c34: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x4f0>
100002c36: 48 8b bd 78 ff ff ff        	movq	-136(%rbp), %rdi
100002c3d: 48 8d 45 80                 	leaq	-128(%rbp), %rax
100002c41: 48 39 c7                    	cmpq	%rax, %rdi
100002c44: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x51e>
100002c46: c5 f8 77                    	vzeroupper
100002c49: e8 02 42 00 00              	callq	16898 <dyld_stub_binder+0x100006e50>
100002c4e: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002c55: 48 85 c0                    	testq	%rax, %rax
100002c58: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x53f>
100002c5a: f0                          	lock
100002c5b: ff 48 14                    	decl	20(%rax)
100002c5e: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x53f>
100002c60: 48 8d bd d0 fe ff ff        	leaq	-304(%rbp), %rdi
100002c67: c5 f8 77                    	vzeroupper
100002c6a: e8 ab 41 00 00              	callq	16811 <dyld_stub_binder+0x100006e1a>
100002c6f: 48 c7 85 08 ff ff ff 00 00 00 00    	movq	$0, -248(%rbp)
100002c7a: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002c7e: c5 fc 11 85 e0 fe ff ff     	vmovups	%ymm0, -288(%rbp)
100002c86: 83 bd d4 fe ff ff 00        	cmpl	$0, -300(%rbp)
100002c8d: 7e 27                       	jle	39 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x586>
100002c8f: 48 8b 85 10 ff ff ff        	movq	-240(%rbp), %rax
100002c96: 31 c9                       	xorl	%ecx, %ecx
100002c98: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002ca0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002ca7: 48 ff c1                    	incq	%rcx
100002caa: 48 63 95 d4 fe ff ff        	movslq	-300(%rbp), %rdx
100002cb1: 48 39 d1                    	cmpq	%rdx, %rcx
100002cb4: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x570>
100002cb6: 48 8b bd 18 ff ff ff        	movq	-232(%rbp), %rdi
100002cbd: 48 8d 85 20 ff ff ff        	leaq	-224(%rbp), %rax
100002cc4: 48 39 c7                    	cmpq	%rax, %rdi
100002cc7: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5a1>
100002cc9: c5 f8 77                    	vzeroupper
100002ccc: e8 7f 41 00 00              	callq	16767 <dyld_stub_binder+0x100006e50>
100002cd1: 48 8b 05 88 63 00 00        	movq	25480(%rip), %rax
100002cd8: 48 8b 00                    	movq	(%rax), %rax
100002cdb: 48 3b 45 d0                 	cmpq	-48(%rbp), %rax
100002cdf: 75 18                       	jne	24 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5c9>
100002ce1: 48 89 d8                    	movq	%rbx, %rax
100002ce4: 48 81 c4 28 01 00 00        	addq	$296, %rsp
100002ceb: 5b                          	popq	%rbx
100002cec: 41 5c                       	popq	%r12
100002cee: 41 5d                       	popq	%r13
100002cf0: 41 5e                       	popq	%r14
100002cf2: 41 5f                       	popq	%r15
100002cf4: 5d                          	popq	%rbp
100002cf5: c5 f8 77                    	vzeroupper
100002cf8: c3                          	retq
100002cf9: c5 f8 77                    	vzeroupper
100002cfc: e8 d3 41 00 00              	callq	16851 <dyld_stub_binder+0x100006ed4>
100002d01: 48 89 c7                    	movq	%rax, %rdi
100002d04: e8 e7 16 00 00              	callq	5863 <_main+0x1510>
100002d09: 48 89 c7                    	movq	%rax, %rdi
100002d0c: e8 df 16 00 00              	callq	5855 <_main+0x1510>
100002d11: eb 1e                       	jmp	30 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x601>
100002d13: eb 00                       	jmp	0 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x5e5>
100002d15: 49 89 c6                    	movq	%rax, %r14
100002d18: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002d1f: 48 85 c0                    	testq	%rax, %rax
100002d22: 0f 85 0f 01 00 00           	jne	271 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x707>
100002d28: e9 1f 01 00 00              	jmp	287 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002d2d: eb 02                       	jmp	2 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x601>
100002d2f: eb 14                       	jmp	20 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x615>
100002d31: 49 89 c6                    	movq	%rax, %r14
100002d34: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002d3b: 48 85 c0                    	testq	%rax, %rax
100002d3e: 75 7f                       	jne	127 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x68f>
100002d40: e9 8f 00 00 00              	jmp	143 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002d45: 49 89 c6                    	movq	%rax, %r14
100002d48: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100002d4c: 48 8b 43 38                 	movq	56(%rbx), %rax
100002d50: 48 85 c0                    	testq	%rax, %rax
100002d53: 74 0e                       	je	14 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x633>
100002d55: f0                          	lock
100002d56: ff 48 14                    	decl	20(%rax)
100002d59: 75 08                       	jne	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x633>
100002d5b: 48 89 df                    	movq	%rbx, %rdi
100002d5e: e8 b7 40 00 00              	callq	16567 <dyld_stub_binder+0x100006e1a>
100002d63: 48 c7 43 38 00 00 00 00     	movq	$0, 56(%rbx)
100002d6b: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002d6f: c5 fc 11 43 10              	vmovups	%ymm0, 16(%rbx)
100002d74: 83 7b 04 00                 	cmpl	$0, 4(%rbx)
100002d78: 7e 20                       	jle	32 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x66a>
100002d7a: 48 8b 4d b0                 	movq	-80(%rbp), %rcx
100002d7e: 48 8d 41 04                 	leaq	4(%rcx), %rax
100002d82: 48 8b 49 40                 	movq	64(%rcx), %rcx
100002d86: 31 d2                       	xorl	%edx, %edx
100002d88: c7 04 91 00 00 00 00        	movl	$0, (%rcx,%rdx,4)
100002d8f: 48 ff c2                    	incq	%rdx
100002d92: 48 63 30                    	movslq	(%rax), %rsi
100002d95: 48 39 f2                    	cmpq	%rsi, %rdx
100002d98: 7c ee                       	jl	-18 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x658>
100002d9a: 48 8b 45 b0                 	movq	-80(%rbp), %rax
100002d9e: 48 8b 78 48                 	movq	72(%rax), %rdi
100002da2: 48 3b bd c8 fe ff ff        	cmpq	-312(%rbp), %rdi
100002da9: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x683>
100002dab: c5 f8 77                    	vzeroupper
100002dae: e8 9d 40 00 00              	callq	16541 <dyld_stub_binder+0x100006e50>
100002db3: 48 8b 85 68 ff ff ff        	movq	-152(%rbp), %rax
100002dba: 48 85 c0                    	testq	%rax, %rax
100002dbd: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002dbf: f0                          	lock
100002dc0: ff 48 14                    	decl	20(%rax)
100002dc3: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6a4>
100002dc5: 48 8d bd 30 ff ff ff        	leaq	-208(%rbp), %rdi
100002dcc: c5 f8 77                    	vzeroupper
100002dcf: e8 46 40 00 00              	callq	16454 <dyld_stub_binder+0x100006e1a>
100002dd4: 48 c7 85 68 ff ff ff 00 00 00 00    	movq	$0, -152(%rbp)
100002ddf: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002de3: c5 fc 11 85 40 ff ff ff     	vmovups	%ymm0, -192(%rbp)
100002deb: 83 bd 34 ff ff ff 00        	cmpl	$0, -204(%rbp)
100002df2: 7e 1f                       	jle	31 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6e3>
100002df4: 48 8b 85 70 ff ff ff        	movq	-144(%rbp), %rax
100002dfb: 31 c9                       	xorl	%ecx, %ecx
100002dfd: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002e04: 48 ff c1                    	incq	%rcx
100002e07: 48 63 95 34 ff ff ff        	movslq	-204(%rbp), %rdx
100002e0e: 48 39 d1                    	cmpq	%rdx, %rcx
100002e11: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6cd>
100002e13: 48 8b bd 78 ff ff ff        	movq	-136(%rbp), %rdi
100002e1a: 48 8d 45 80                 	leaq	-128(%rbp), %rax
100002e1e: 48 39 c7                    	cmpq	%rax, %rdi
100002e21: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x6fb>
100002e23: c5 f8 77                    	vzeroupper
100002e26: e8 25 40 00 00              	callq	16421 <dyld_stub_binder+0x100006e50>
100002e2b: 48 8b 85 08 ff ff ff        	movq	-248(%rbp), %rax
100002e32: 48 85 c0                    	testq	%rax, %rax
100002e35: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002e37: f0                          	lock
100002e38: ff 48 14                    	decl	20(%rax)
100002e3b: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x71c>
100002e3d: 48 8d bd d0 fe ff ff        	leaq	-304(%rbp), %rdi
100002e44: c5 f8 77                    	vzeroupper
100002e47: e8 ce 3f 00 00              	callq	16334 <dyld_stub_binder+0x100006e1a>
100002e4c: 48 c7 85 08 ff ff ff 00 00 00 00    	movq	$0, -248(%rbp)
100002e57: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100002e5b: c5 fc 11 85 e0 fe ff ff     	vmovups	%ymm0, -288(%rbp)
100002e63: 83 bd d4 fe ff ff 00        	cmpl	$0, -300(%rbp)
100002e6a: 7e 2a                       	jle	42 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x766>
100002e6c: 48 8b 85 10 ff ff ff        	movq	-240(%rbp), %rax
100002e73: 31 c9                       	xorl	%ecx, %ecx
100002e75: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002e7f: 90                          	nop
100002e80: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002e87: 48 ff c1                    	incq	%rcx
100002e8a: 48 63 95 d4 fe ff ff        	movslq	-300(%rbp), %rdx
100002e91: 48 39 d1                    	cmpq	%rdx, %rcx
100002e94: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x750>
100002e96: 48 8b bd 18 ff ff ff        	movq	-232(%rbp), %rdi
100002e9d: 48 8d 85 20 ff ff ff        	leaq	-224(%rbp), %rax
100002ea4: 48 39 c7                    	cmpq	%rax, %rdi
100002ea7: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER14ModelInterfacef+0x781>
100002ea9: c5 f8 77                    	vzeroupper
100002eac: e8 9f 3f 00 00              	callq	16287 <dyld_stub_binder+0x100006e50>
100002eb1: 4c 89 f7                    	movq	%r14, %rdi
100002eb4: c5 f8 77                    	vzeroupper
100002eb7: e8 46 3f 00 00              	callq	16198 <dyld_stub_binder+0x100006e02>
100002ebc: 0f 0b                       	ud2
100002ebe: 48 89 c7                    	movq	%rax, %rdi
100002ec1: e8 2a 15 00 00              	callq	5418 <_main+0x1510>
100002ec6: 48 89 c7                    	movq	%rax, %rdi
100002ec9: e8 22 15 00 00              	callq	5410 <_main+0x1510>
100002ece: 48 89 c7                    	movq	%rax, %rdi
100002ed1: e8 1a 15 00 00              	callq	5402 <_main+0x1510>
100002ed6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

0000000100002ee0 _main:
100002ee0: 55                          	pushq	%rbp
100002ee1: 48 89 e5                    	movq	%rsp, %rbp
100002ee4: 41 57                       	pushq	%r15
100002ee6: 41 56                       	pushq	%r14
100002ee8: 41 55                       	pushq	%r13
100002eea: 41 54                       	pushq	%r12
100002eec: 53                          	pushq	%rbx
100002eed: 48 83 e4 e0                 	andq	$-32, %rsp
100002ef1: 48 81 ec 00 04 00 00        	subq	$1024, %rsp
100002ef8: 48 8b 05 61 61 00 00        	movq	24929(%rip), %rax
100002eff: 48 8b 00                    	movq	(%rax), %rax
100002f02: 48 89 84 24 e0 03 00 00     	movq	%rax, 992(%rsp)
100002f0a: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100002f12: e8 d9 27 00 00              	callq	10201 <__ZN11LineNetworkC1Ev>
100002f17: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002f1b: c5 f9 7f 84 24 60 02 00 00  	vmovdqa	%xmm0, 608(%rsp)
100002f24: 48 c7 84 24 70 02 00 00 00 00 00 00 	movq	$0, 624(%rsp)
100002f30: bf 30 00 00 00              	movl	$48, %edi
100002f35: e8 88 3f 00 00              	callq	16264 <dyld_stub_binder+0x100006ec2>
100002f3a: 48 89 84 24 70 02 00 00     	movq	%rax, 624(%rsp)
100002f42: c5 f8 28 05 86 41 00 00     	vmovaps	16774(%rip), %xmm0
100002f4a: c5 f8 29 84 24 60 02 00 00  	vmovaps	%xmm0, 608(%rsp)
100002f53: c5 fe 6f 05 59 5f 00 00     	vmovdqu	24409(%rip), %ymm0
100002f5b: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002f5f: 48 b9 69 64 65 6f 2e 6d 70 34       	movabsq	$3778640133568685161, %rcx
100002f69: 48 89 48 20                 	movq	%rcx, 32(%rax)
100002f6d: c6 40 28 00                 	movb	$0, 40(%rax)
100002f71: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100002f79: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
100002f81: 31 d2                       	xorl	%edx, %edx
100002f83: c5 f8 77                    	vzeroupper
100002f86: e8 7d 3e 00 00              	callq	15997 <dyld_stub_binder+0x100006e08>
100002f8b: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100002f93: 74 0d                       	je	13 <_main+0xc2>
100002f95: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
100002f9d: e8 14 3f 00 00              	callq	16148 <dyld_stub_binder+0x100006eb6>
100002fa2: 4c 8d 74 24 68              	leaq	104(%rsp), %r14
100002fa7: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002fab: c5 f9 d6 84 24 d8 00 00 00  	vmovq	%xmm0, 216(%rsp)
100002fb4: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100002fbc: 4c 8d bc 24 c0 03 00 00     	leaq	960(%rsp), %r15
100002fc4: 4c 8d ac 24 c0 01 00 00     	leaq	448(%rsp), %r13
100002fcc: eb 0b                       	jmp	11 <_main+0xf9>
100002fce: 66 90                       	nop
100002fd0: 45 85 e4                    	testl	%r12d, %r12d
100002fd3: 0f 85 a1 0f 00 00           	jne	4001 <_main+0x109a>
100002fd9: 48 89 df                    	movq	%rbx, %rdi
100002fdc: c5 f8 77                    	vzeroupper
100002fdf: e8 78 3e 00 00              	callq	15992 <dyld_stub_binder+0x100006e5c>
100002fe4: 84 c0                       	testb	%al, %al
100002fe6: 0f 84 8e 0f 00 00           	je	3982 <_main+0x109a>
100002fec: c7 44 24 18 00 00 ff 42     	movl	$1124007936, 24(%rsp)
100002ff4: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002ff8: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
100002ffd: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
100003002: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100003006: 48 8d 44 24 20              	leaq	32(%rsp), %rax
10000300b: 48 89 44 24 58              	movq	%rax, 88(%rsp)
100003010: 4c 89 74 24 60              	movq	%r14, 96(%rsp)
100003015: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003019: c4 c1 7a 7f 06              	vmovdqu	%xmm0, (%r14)
10000301e: 48 89 df                    	movq	%rbx, %rdi
100003021: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100003026: c5 f8 77                    	vzeroupper
100003029: e8 e6 3d 00 00              	callq	15846 <dyld_stub_binder+0x100006e14>
10000302e: 41 bc 03 00 00 00           	movl	$3, %r12d
100003034: 48 83 7c 24 28 00           	cmpq	$0, 40(%rsp)
10000303a: 0f 84 80 08 00 00           	je	2176 <_main+0x9e0>
100003040: 8b 44 24 1c                 	movl	28(%rsp), %eax
100003044: 83 f8 03                    	cmpl	$3, %eax
100003047: 0f 8d 53 03 00 00           	jge	851 <_main+0x4c0>
10000304d: 48 63 4c 24 20              	movslq	32(%rsp), %rcx
100003052: 48 63 74 24 24              	movslq	36(%rsp), %rsi
100003057: 48 0f af f1                 	imulq	%rcx, %rsi
10000305b: 85 c0                       	testl	%eax, %eax
10000305d: 0f 84 5d 08 00 00           	je	2141 <_main+0x9e0>
100003063: 48 85 f6                    	testq	%rsi, %rsi
100003066: 0f 84 54 08 00 00           	je	2132 <_main+0x9e0>
10000306c: bf 19 00 00 00              	movl	$25, %edi
100003071: c5 f8 77                    	vzeroupper
100003074: e8 cb 3d 00 00              	callq	15819 <dyld_stub_binder+0x100006e44>
100003079: 3c 1b                       	cmpb	$27, %al
10000307b: 0f 84 3f 08 00 00           	je	2111 <_main+0x9e0>
100003081: e8 06 3e 00 00              	callq	15878 <dyld_stub_binder+0x100006e8c>
100003086: 49 89 c6                    	movq	%rax, %r14
100003089: 48 8d 9c 24 00 01 00 00     	leaq	256(%rsp), %rbx
100003091: 48 89 df                    	movq	%rbx, %rdi
100003094: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100003099: 48 8d 94 24 08 02 00 00     	leaq	520(%rsp), %rdx
1000030a1: c5 f9 6e 05 03 40 00 00     	vmovd	16387(%rip), %xmm0
1000030a9: e8 82 f6 ff ff              	callq	-2430 <__Z14get_predictionRN2cv3MatER14ModelInterfacef>
1000030ae: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
1000030b6: c5 fa 7e 05 b2 3f 00 00     	vmovq	16306(%rip), %xmm0
1000030be: 48 89 de                    	movq	%rbx, %rsi
1000030c1: e8 90 3d 00 00              	callq	15760 <dyld_stub_binder+0x100006e56>
1000030c6: 48 8b 84 24 38 01 00 00     	movq	312(%rsp), %rax
1000030ce: 48 85 c0                    	testq	%rax, %rax
1000030d1: 74 0e                       	je	14 <_main+0x201>
1000030d3: f0                          	lock
1000030d4: ff 48 14                    	decl	20(%rax)
1000030d7: 75 08                       	jne	8 <_main+0x201>
1000030d9: 48 89 df                    	movq	%rbx, %rdi
1000030dc: e8 39 3d 00 00              	callq	15673 <dyld_stub_binder+0x100006e1a>
1000030e1: 48 c7 84 24 38 01 00 00 00 00 00 00 	movq	$0, 312(%rsp)
1000030ed: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
1000030f5: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000030f9: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
1000030fd: 83 bc 24 04 01 00 00 00     	cmpl	$0, 260(%rsp)
100003105: 7e 30                       	jle	48 <_main+0x257>
100003107: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
10000310f: 31 c9                       	xorl	%ecx, %ecx
100003111: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000311b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003120: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003127: 48 ff c1                    	incq	%rcx
10000312a: 48 63 94 24 04 01 00 00     	movslq	260(%rsp), %rdx
100003132: 48 39 d1                    	cmpq	%rdx, %rcx
100003135: 7c e9                       	jl	-23 <_main+0x240>
100003137: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
10000313f: 48 8d 84 24 50 01 00 00     	leaq	336(%rsp), %rax
100003147: 48 39 c7                    	cmpq	%rax, %rdi
10000314a: 74 08                       	je	8 <_main+0x274>
10000314c: c5 f8 77                    	vzeroupper
10000314f: e8 fc 3c 00 00              	callq	15612 <dyld_stub_binder+0x100006e50>
100003154: c5 f8 77                    	vzeroupper
100003157: e8 30 3d 00 00              	callq	15664 <dyld_stub_binder+0x100006e8c>
10000315c: 49 89 c4                    	movq	%rax, %r12
10000315f: c7 84 24 00 01 00 00 00 00 ff 42    	movl	$1124007936, 256(%rsp)
10000316a: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
100003172: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003176: c5 fe 7f 40 f4              	vmovdqu	%ymm0, -12(%rax)
10000317b: c5 fe 7f 40 10              	vmovdqu	%ymm0, 16(%rax)
100003180: 48 8b 44 24 20              	movq	32(%rsp), %rax
100003185: 48 8d 8c 24 08 01 00 00     	leaq	264(%rsp), %rcx
10000318d: 48 89 8c 24 40 01 00 00     	movq	%rcx, 320(%rsp)
100003195: 48 8d 8c 24 50 01 00 00     	leaq	336(%rsp), %rcx
10000319d: 48 89 8c 24 48 01 00 00     	movq	%rcx, 328(%rsp)
1000031a5: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000031a9: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
1000031ad: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
1000031b5: 48 89 df                    	movq	%rbx, %rdi
1000031b8: be 02 00 00 00              	movl	$2, %esi
1000031bd: 4c 89 fa                    	movq	%r15, %rdx
1000031c0: 31 c9                       	xorl	%ecx, %ecx
1000031c2: c5 f8 77                    	vzeroupper
1000031c5: e8 56 3c 00 00              	callq	15446 <dyld_stub_binder+0x100006e20>
1000031ca: 48 c7 84 24 88 00 00 00 00 00 00 00 	movq	$0, 136(%rsp)
1000031d6: c7 44 24 78 00 00 06 c1     	movl	$3238395904, 120(%rsp)
1000031de: 48 8d 84 24 60 02 00 00     	leaq	608(%rsp), %rax
1000031e6: 48 89 84 24 80 00 00 00     	movq	%rax, 128(%rsp)
1000031ee: 48 c7 84 24 70 01 00 00 00 00 00 00 	movq	$0, 368(%rsp)
1000031fa: c7 84 24 60 01 00 00 00 00 01 02    	movl	$33619968, 352(%rsp)
100003205: 48 89 9c 24 68 01 00 00     	movq	%rbx, 360(%rsp)
10000320d: 8b 44 24 20                 	movl	32(%rsp), %eax
100003211: 8b 4c 24 24                 	movl	36(%rsp), %ecx
100003215: 89 8c 24 b0 01 00 00        	movl	%ecx, 432(%rsp)
10000321c: 89 84 24 b4 01 00 00        	movl	%eax, 436(%rsp)
100003223: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003227: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
10000322b: 48 8d 5c 24 78              	leaq	120(%rsp), %rbx
100003230: 48 89 df                    	movq	%rbx, %rdi
100003233: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
10000323b: 48 8d 94 24 b0 01 00 00     	leaq	432(%rsp), %rdx
100003243: b9 01 00 00 00              	movl	$1, %ecx
100003248: e8 eb 3b 00 00              	callq	15339 <dyld_stub_binder+0x100006e38>
10000324d: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003251: c5 fd 7f 84 24 60 01 00 00  	vmovdqa	%ymm0, 352(%rsp)
10000325a: c7 44 24 78 00 00 ff 42     	movl	$1124007936, 120(%rsp)
100003262: 48 8d 44 24 7c              	leaq	124(%rsp), %rax
100003267: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
10000326c: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100003270: 48 8b 44 24 20              	movq	32(%rsp), %rax
100003275: 48 8d 8c 24 80 00 00 00     	leaq	128(%rsp), %rcx
10000327d: 48 89 8c 24 b8 00 00 00     	movq	%rcx, 184(%rsp)
100003285: 48 8d 8c 24 c8 00 00 00     	leaq	200(%rsp), %rcx
10000328d: 48 89 8c 24 c0 00 00 00     	movq	%rcx, 192(%rsp)
100003295: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003299: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
10000329d: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
1000032a5: 48 89 df                    	movq	%rbx, %rdi
1000032a8: be 02 00 00 00              	movl	$2, %esi
1000032ad: 4c 89 fa                    	movq	%r15, %rdx
1000032b0: b9 10 00 00 00              	movl	$16, %ecx
1000032b5: c5 f8 77                    	vzeroupper
1000032b8: e8 63 3b 00 00              	callq	15203 <dyld_stub_binder+0x100006e20>
1000032bd: 48 89 df                    	movq	%rbx, %rdi
1000032c0: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
1000032c8: e8 5f 3b 00 00              	callq	15199 <dyld_stub_binder+0x100006e2c>
1000032cd: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000032d2: 48 85 c0                    	testq	%rax, %rax
1000032d5: 74 04                       	je	4 <_main+0x3fb>
1000032d7: f0                          	lock
1000032d8: ff 40 14                    	incl	20(%rax)
1000032db: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
1000032e3: 48 85 c0                    	testq	%rax, %rax
1000032e6: 74 10                       	je	16 <_main+0x418>
1000032e8: f0                          	lock
1000032e9: ff 48 14                    	decl	20(%rax)
1000032ec: 75 0a                       	jne	10 <_main+0x418>
1000032ee: 48 8d 7c 24 78              	leaq	120(%rsp), %rdi
1000032f3: e8 22 3b 00 00              	callq	15138 <dyld_stub_binder+0x100006e1a>
1000032f8: 48 c7 84 24 b0 00 00 00 00 00 00 00 	movq	$0, 176(%rsp)
100003304: 48 8d 44 24 7c              	leaq	124(%rsp), %rax
100003309: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000330d: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100003312: 83 7c 24 7c 00              	cmpl	$0, 124(%rsp)
100003317: 0f 8e 22 06 00 00           	jle	1570 <_main+0xa5f>
10000331d: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003325: 31 c9                       	xorl	%ecx, %ecx
100003327: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100003330: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003337: 48 ff c1                    	incq	%rcx
10000333a: 48 63 54 24 7c              	movslq	124(%rsp), %rdx
10000333f: 48 39 d1                    	cmpq	%rdx, %rcx
100003342: 7c ec                       	jl	-20 <_main+0x450>
100003344: 8b 44 24 18                 	movl	24(%rsp), %eax
100003348: 89 44 24 78                 	movl	%eax, 120(%rsp)
10000334c: 83 fa 02                    	cmpl	$2, %edx
10000334f: 0f 8f ff 05 00 00           	jg	1535 <_main+0xa74>
100003355: 8b 44 24 1c                 	movl	28(%rsp), %eax
100003359: 83 f8 02                    	cmpl	$2, %eax
10000335c: 0f 8f f2 05 00 00           	jg	1522 <_main+0xa74>
100003362: 89 44 24 7c                 	movl	%eax, 124(%rsp)
100003366: 8b 4c 24 20                 	movl	32(%rsp), %ecx
10000336a: 8b 44 24 24                 	movl	36(%rsp), %eax
10000336e: 89 8c 24 80 00 00 00        	movl	%ecx, 128(%rsp)
100003375: 89 84 24 84 00 00 00        	movl	%eax, 132(%rsp)
10000337c: 48 8b 44 24 60              	movq	96(%rsp), %rax
100003381: 48 8b 10                    	movq	(%rax), %rdx
100003384: 48 8b b4 24 c0 00 00 00     	movq	192(%rsp), %rsi
10000338c: 48 89 16                    	movq	%rdx, (%rsi)
10000338f: 48 8b 40 08                 	movq	8(%rax), %rax
100003393: 48 89 46 08                 	movq	%rax, 8(%rsi)
100003397: e9 ce 05 00 00              	jmp	1486 <_main+0xa8a>
10000339c: 0f 1f 40 00                 	nopl	(%rax)
1000033a0: 48 8b 4c 24 58              	movq	88(%rsp), %rcx
1000033a5: 83 f8 0f                    	cmpl	$15, %eax
1000033a8: 77 0c                       	ja	12 <_main+0x4d6>
1000033aa: be 01 00 00 00              	movl	$1, %esi
1000033af: 31 d2                       	xorl	%edx, %edx
1000033b1: e9 ea 04 00 00              	jmp	1258 <_main+0x9c0>
1000033b6: 89 c2                       	movl	%eax, %edx
1000033b8: 83 e2 f0                    	andl	$-16, %edx
1000033bb: 48 8d 72 f0                 	leaq	-16(%rdx), %rsi
1000033bf: 48 89 f7                    	movq	%rsi, %rdi
1000033c2: 48 c1 ef 04                 	shrq	$4, %rdi
1000033c6: 48 ff c7                    	incq	%rdi
1000033c9: 89 fb                       	movl	%edi, %ebx
1000033cb: 83 e3 03                    	andl	$3, %ebx
1000033ce: 48 83 fe 30                 	cmpq	$48, %rsi
1000033d2: 73 25                       	jae	37 <_main+0x519>
1000033d4: c4 e2 7d 59 05 8b 3c 00 00  	vpbroadcastq	15499(%rip), %ymm0
1000033dd: 31 ff                       	xorl	%edi, %edi
1000033df: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
1000033e3: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
1000033e7: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
1000033eb: 48 85 db                    	testq	%rbx, %rbx
1000033ee: 0f 85 0e 03 00 00           	jne	782 <_main+0x822>
1000033f4: e9 d0 03 00 00              	jmp	976 <_main+0x8e9>
1000033f9: 48 89 de                    	movq	%rbx, %rsi
1000033fc: 48 29 fe                    	subq	%rdi, %rsi
1000033ff: c4 e2 7d 59 05 60 3c 00 00  	vpbroadcastq	15456(%rip), %ymm0
100003408: 31 ff                       	xorl	%edi, %edi
10000340a: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
10000340e: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100003412: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
100003416: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003420: c4 e2 7d 25 24 b9           	vpmovsxdq	(%rcx,%rdi,4), %ymm4
100003426: c4 e2 7d 25 6c b9 10        	vpmovsxdq	16(%rcx,%rdi,4), %ymm5
10000342d: c4 e2 7d 25 74 b9 20        	vpmovsxdq	32(%rcx,%rdi,4), %ymm6
100003434: c4 e2 7d 25 7c b9 30        	vpmovsxdq	48(%rcx,%rdi,4), %ymm7
10000343b: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
100003440: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
100003444: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
100003449: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
10000344e: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
100003453: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003459: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000345d: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003462: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
100003467: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
10000346b: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
100003470: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
100003475: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
100003479: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000347e: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003482: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100003486: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
10000348b: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
10000348f: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100003494: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100003498: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000349c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000034a1: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
1000034a5: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000034a9: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
1000034ae: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
1000034b2: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
1000034b7: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
1000034bb: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
1000034bf: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000034c4: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
1000034c8: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000034cc: c4 e2 7d 25 64 b9 40        	vpmovsxdq	64(%rcx,%rdi,4), %ymm4
1000034d3: c4 e2 7d 25 6c b9 50        	vpmovsxdq	80(%rcx,%rdi,4), %ymm5
1000034da: c4 e2 7d 25 74 b9 60        	vpmovsxdq	96(%rcx,%rdi,4), %ymm6
1000034e1: c4 e2 7d 25 7c b9 70        	vpmovsxdq	112(%rcx,%rdi,4), %ymm7
1000034e8: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
1000034ed: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
1000034f2: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
1000034f7: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
1000034fb: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100003500: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003506: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000350a: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
10000350f: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100003514: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100003518: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
10000351d: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100003521: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100003526: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000352b: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
10000352f: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100003533: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100003538: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
10000353c: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100003541: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100003545: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100003549: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000354e: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003552: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003556: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
10000355b: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
10000355f: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100003564: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100003568: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
10000356c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003571: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100003575: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100003579: c4 e2 7d 25 a4 b9 80 00 00 00       	vpmovsxdq	128(%rcx,%rdi,4), %ymm4
100003583: c4 e2 7d 25 ac b9 90 00 00 00       	vpmovsxdq	144(%rcx,%rdi,4), %ymm5
10000358d: c4 e2 7d 25 b4 b9 a0 00 00 00       	vpmovsxdq	160(%rcx,%rdi,4), %ymm6
100003597: c4 e2 7d 25 bc b9 b0 00 00 00       	vpmovsxdq	176(%rcx,%rdi,4), %ymm7
1000035a1: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
1000035a6: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
1000035ab: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
1000035b0: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
1000035b4: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
1000035b9: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000035bf: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000035c3: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000035c8: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
1000035cd: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
1000035d1: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
1000035d6: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
1000035da: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
1000035df: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000035e4: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000035e8: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000035ec: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
1000035f1: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
1000035f5: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
1000035fa: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
1000035fe: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100003602: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003607: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
10000360b: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
10000360f: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
100003614: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100003618: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
10000361d: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100003621: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100003625: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000362a: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
10000362e: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100003632: c4 e2 7d 25 a4 b9 c0 00 00 00       	vpmovsxdq	192(%rcx,%rdi,4), %ymm4
10000363c: c4 e2 7d 25 ac b9 d0 00 00 00       	vpmovsxdq	208(%rcx,%rdi,4), %ymm5
100003646: c4 e2 7d 25 b4 b9 e0 00 00 00       	vpmovsxdq	224(%rcx,%rdi,4), %ymm6
100003650: c4 e2 7d 25 bc b9 f0 00 00 00       	vpmovsxdq	240(%rcx,%rdi,4), %ymm7
10000365a: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
10000365f: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100003664: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100003669: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
10000366d: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100003672: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003678: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000367c: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003681: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100003686: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
10000368a: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
10000368f: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100003693: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100003698: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000369d: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000036a1: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000036a5: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
1000036aa: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
1000036ae: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
1000036b3: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
1000036b7: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000036bb: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000036c0: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
1000036c4: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000036c8: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
1000036cd: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
1000036d1: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
1000036d6: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
1000036da: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000036de: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000036e3: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
1000036e7: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000036eb: 48 83 c7 40                 	addq	$64, %rdi
1000036ef: 48 83 c6 04                 	addq	$4, %rsi
1000036f3: 0f 85 27 fd ff ff           	jne	-729 <_main+0x540>
1000036f9: 48 85 db                    	testq	%rbx, %rbx
1000036fc: 0f 84 c7 00 00 00           	je	199 <_main+0x8e9>
100003702: 48 8d 34 b9                 	leaq	(%rcx,%rdi,4), %rsi
100003706: 48 83 c6 30                 	addq	$48, %rsi
10000370a: 48 c1 e3 06                 	shlq	$6, %rbx
10000370e: 31 ff                       	xorl	%edi, %edi
100003710: c4 e2 7d 25 64 3e d0        	vpmovsxdq	-48(%rsi,%rdi), %ymm4
100003717: c4 e2 7d 25 6c 3e e0        	vpmovsxdq	-32(%rsi,%rdi), %ymm5
10000371e: c4 e2 7d 25 74 3e f0        	vpmovsxdq	-16(%rsi,%rdi), %ymm6
100003725: c4 e2 7d 25 3c 3e           	vpmovsxdq	(%rsi,%rdi), %ymm7
10000372b: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
100003730: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
100003734: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
100003739: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
10000373e: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
100003743: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100003749: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000374d: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100003752: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
100003757: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
10000375b: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
100003760: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
100003765: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
100003769: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000376e: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100003772: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100003776: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
10000377b: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
10000377f: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100003784: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100003788: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000378c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100003791: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100003795: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100003799: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
10000379e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
1000037a2: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
1000037a7: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
1000037ab: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
1000037af: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000037b4: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
1000037b8: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000037bc: 48 83 c7 40                 	addq	$64, %rdi
1000037c0: 48 39 fb                    	cmpq	%rdi, %rbx
1000037c3: 0f 85 47 ff ff ff           	jne	-185 <_main+0x830>
1000037c9: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
1000037ce: c5 dd f4 e0                 	vpmuludq	%ymm0, %ymm4, %ymm4
1000037d2: c5 d5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm5
1000037d7: c5 e5 f4 ed                 	vpmuludq	%ymm5, %ymm3, %ymm5
1000037db: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
1000037df: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000037e4: c5 e5 f4 c0                 	vpmuludq	%ymm0, %ymm3, %ymm0
1000037e8: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
1000037ec: c5 e5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm3
1000037f1: c5 e5 f4 d8                 	vpmuludq	%ymm0, %ymm3, %ymm3
1000037f5: c5 dd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm4
1000037fa: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
1000037fe: c5 dd d4 db                 	vpaddq	%ymm3, %ymm4, %ymm3
100003802: c5 e5 73 f3 20              	vpsllq	$32, %ymm3, %ymm3
100003807: c5 ed f4 c0                 	vpmuludq	%ymm0, %ymm2, %ymm0
10000380b: c5 fd d4 c3                 	vpaddq	%ymm3, %ymm0, %ymm0
10000380f: c5 ed 73 d1 20              	vpsrlq	$32, %ymm1, %ymm2
100003814: c5 ed f4 d0                 	vpmuludq	%ymm0, %ymm2, %ymm2
100003818: c5 e5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm3
10000381d: c5 f5 f4 db                 	vpmuludq	%ymm3, %ymm1, %ymm3
100003821: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
100003825: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
10000382a: c5 f5 f4 c0                 	vpmuludq	%ymm0, %ymm1, %ymm0
10000382e: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
100003832: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100003838: c5 ed 73 d0 20              	vpsrlq	$32, %ymm0, %ymm2
10000383d: c5 ed f4 d1                 	vpmuludq	%ymm1, %ymm2, %ymm2
100003841: c5 e5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm3
100003846: c5 fd f4 db                 	vpmuludq	%ymm3, %ymm0, %ymm3
10000384a: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
10000384e: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
100003853: c5 fd f4 c1                 	vpmuludq	%ymm1, %ymm0, %ymm0
100003857: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
10000385b: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100003860: c5 e9 73 d0 20              	vpsrlq	$32, %xmm0, %xmm2
100003865: c5 e9 f4 d1                 	vpmuludq	%xmm1, %xmm2, %xmm2
100003869: c5 e1 73 d8 0c              	vpsrldq	$12, %xmm0, %xmm3
10000386e: c5 f9 f4 db                 	vpmuludq	%xmm3, %xmm0, %xmm3
100003872: c5 e1 d4 d2                 	vpaddq	%xmm2, %xmm3, %xmm2
100003876: c5 e9 73 f2 20              	vpsllq	$32, %xmm2, %xmm2
10000387b: c5 f9 f4 c1                 	vpmuludq	%xmm1, %xmm0, %xmm0
10000387f: c5 f9 d4 c2                 	vpaddq	%xmm2, %xmm0, %xmm0
100003883: c4 e1 f9 7e c6              	vmovq	%xmm0, %rsi
100003888: 48 39 c2                    	cmpq	%rax, %rdx
10000388b: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100003893: 74 1b                       	je	27 <_main+0x9d0>
100003895: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000389f: 90                          	nop
1000038a0: 48 63 3c 91                 	movslq	(%rcx,%rdx,4), %rdi
1000038a4: 48 0f af f7                 	imulq	%rdi, %rsi
1000038a8: 48 ff c2                    	incq	%rdx
1000038ab: 48 39 d0                    	cmpq	%rdx, %rax
1000038ae: 75 f0                       	jne	-16 <_main+0x9c0>
1000038b0: 85 c0                       	testl	%eax, %eax
1000038b2: 0f 85 ab f7 ff ff           	jne	-2133 <_main+0x183>
1000038b8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
1000038c0: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000038c5: 48 85 c0                    	testq	%rax, %rax
1000038c8: 74 13                       	je	19 <_main+0x9fd>
1000038ca: f0                          	lock
1000038cb: ff 48 14                    	decl	20(%rax)
1000038ce: 75 0d                       	jne	13 <_main+0x9fd>
1000038d0: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
1000038d5: c5 f8 77                    	vzeroupper
1000038d8: e8 3d 35 00 00              	callq	13629 <dyld_stub_binder+0x100006e1a>
1000038dd: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
1000038e6: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000038ea: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
1000038ef: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
1000038f4: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
1000038f9: 7e 29                       	jle	41 <_main+0xa44>
1000038fb: 48 8b 44 24 58              	movq	88(%rsp), %rax
100003900: 31 c9                       	xorl	%ecx, %ecx
100003902: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000390c: 0f 1f 40 00                 	nopl	(%rax)
100003910: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003917: 48 ff c1                    	incq	%rcx
10000391a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000391f: 48 39 d1                    	cmpq	%rdx, %rcx
100003922: 7c ec                       	jl	-20 <_main+0xa30>
100003924: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003929: 4c 39 f7                    	cmpq	%r14, %rdi
10000392c: 0f 84 9e f6 ff ff           	je	-2402 <_main+0xf0>
100003932: c5 f8 77                    	vzeroupper
100003935: e8 16 35 00 00              	callq	13590 <dyld_stub_binder+0x100006e50>
10000393a: e9 91 f6 ff ff              	jmp	-2415 <_main+0xf0>
10000393f: 8b 44 24 18                 	movl	24(%rsp), %eax
100003943: 89 44 24 78                 	movl	%eax, 120(%rsp)
100003947: 8b 44 24 1c                 	movl	28(%rsp), %eax
10000394b: 83 f8 02                    	cmpl	$2, %eax
10000394e: 0f 8e 0e fa ff ff           	jle	-1522 <_main+0x482>
100003954: 48 8d 7c 24 78              	leaq	120(%rsp), %rdi
100003959: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
10000395e: c5 f8 77                    	vzeroupper
100003961: e8 c0 34 00 00              	callq	13504 <dyld_stub_binder+0x100006e26>
100003966: 8b 4c 24 20                 	movl	32(%rsp), %ecx
10000396a: c4 c1 eb 2a c6              	vcvtsi2sd	%r14, %xmm2, %xmm0
10000396f: c4 c1 eb 2a cc              	vcvtsi2sd	%r12, %xmm2, %xmm1
100003974: c5 fb 10 15 e4 36 00 00     	vmovsd	14052(%rip), %xmm2
10000397c: c5 fb 5e c2                 	vdivsd	%xmm2, %xmm0, %xmm0
100003980: c5 f3 5e ca                 	vdivsd	%xmm2, %xmm1, %xmm1
100003984: c5 fc 10 54 24 28           	vmovups	40(%rsp), %ymm2
10000398a: c5 fc 11 94 24 88 00 00 00  	vmovups	%ymm2, 136(%rsp)
100003993: c5 f9 10 54 24 48           	vmovupd	72(%rsp), %xmm2
100003999: c5 f9 11 94 24 a8 00 00 00  	vmovupd	%xmm2, 168(%rsp)
1000039a2: 85 c9                       	testl	%ecx, %ecx
1000039a4: 4d 89 fe                    	movq	%r15, %r14
1000039a7: 0f 84 49 01 00 00           	je	329 <_main+0xc16>
1000039ad: 31 c0                       	xorl	%eax, %eax
1000039af: 8b 74 24 24                 	movl	36(%rsp), %esi
1000039b3: 85 f6                       	testl	%esi, %esi
1000039b5: be 00 00 00 00              	movl	$0, %esi
1000039ba: 75 17                       	jne	23 <_main+0xaf3>
1000039bc: 0f 1f 40 00                 	nopl	(%rax)
1000039c0: ff c0                       	incl	%eax
1000039c2: 39 c8                       	cmpl	%ecx, %eax
1000039c4: 0f 83 2c 01 00 00           	jae	300 <_main+0xc16>
1000039ca: 85 f6                       	testl	%esi, %esi
1000039cc: be 00 00 00 00              	movl	$0, %esi
1000039d1: 74 ed                       	je	-19 <_main+0xae0>
1000039d3: 48 63 c8                    	movslq	%eax, %rcx
1000039d6: 31 d2                       	xorl	%edx, %edx
1000039d8: c5 fb 10 25 98 36 00 00     	vmovsd	13976(%rip), %xmm4
1000039e0: c5 fa 10 2d c8 36 00 00     	vmovss	14024(%rip), %xmm5
1000039e8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
1000039f0: 48 8b 74 24 60              	movq	96(%rsp), %rsi
1000039f5: 48 8b 3e                    	movq	(%rsi), %rdi
1000039f8: 48 0f af f9                 	imulq	%rcx, %rdi
1000039fc: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100003a01: 48 63 d2                    	movslq	%edx, %rdx
100003a04: 48 8d 34 52                 	leaq	(%rdx,%rdx,2), %rsi
100003a08: 0f b6 3c 37                 	movzbl	(%rdi,%rsi), %edi
100003a0c: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003a10: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003a14: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003a18: 48 8b 9c 24 c0 00 00 00     	movq	192(%rsp), %rbx
100003a20: 48 8b 1b                    	movq	(%rbx), %rbx
100003a23: 48 0f af d9                 	imulq	%rcx, %rbx
100003a27: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100003a2f: 40 88 3c 33                 	movb	%dil, (%rbx,%rsi)
100003a33: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003a38: 48 8b 3f                    	movq	(%rdi), %rdi
100003a3b: 48 0f af f9                 	imulq	%rcx, %rdi
100003a3f: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100003a44: 0f b6 7c 37 01              	movzbl	1(%rdi,%rsi), %edi
100003a49: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003a4d: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
100003a55: 48 8b 3f                    	movq	(%rdi), %rdi
100003a58: 48 0f af f9                 	imulq	%rcx, %rdi
100003a5c: 48 03 bc 24 10 01 00 00     	addq	272(%rsp), %rdi
100003a64: 0f b6 3c 3a                 	movzbl	(%rdx,%rdi), %edi
100003a68: c5 ca 2a df                 	vcvtsi2ss	%edi, %xmm6, %xmm3
100003a6c: c5 e2 59 dd                 	vmulss	%xmm5, %xmm3, %xmm3
100003a70: c5 e2 5a db                 	vcvtss2sd	%xmm3, %xmm3, %xmm3
100003a74: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003a78: c5 eb 58 d3                 	vaddsd	%xmm3, %xmm2, %xmm2
100003a7c: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003a80: 48 8b 9c 24 c0 00 00 00     	movq	192(%rsp), %rbx
100003a88: 48 8b 1b                    	movq	(%rbx), %rbx
100003a8b: 48 0f af d9                 	imulq	%rcx, %rbx
100003a8f: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100003a97: 40 88 7c 33 01              	movb	%dil, 1(%rbx,%rsi)
100003a9c: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003aa1: 48 8b 3f                    	movq	(%rdi), %rdi
100003aa4: 48 0f af f9                 	imulq	%rcx, %rdi
100003aa8: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100003aad: 0f b6 7c 37 02              	movzbl	2(%rdi,%rsi), %edi
100003ab2: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100003ab6: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100003aba: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100003abe: 48 8b 9c 24 c0 00 00 00     	movq	192(%rsp), %rbx
100003ac6: 48 8b 1b                    	movq	(%rbx), %rbx
100003ac9: 48 0f af d9                 	imulq	%rcx, %rbx
100003acd: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100003ad5: 40 88 7c 33 02              	movb	%dil, 2(%rbx,%rsi)
100003ada: ff c2                       	incl	%edx
100003adc: 8b 74 24 24                 	movl	36(%rsp), %esi
100003ae0: 39 f2                       	cmpl	%esi, %edx
100003ae2: 0f 82 08 ff ff ff           	jb	-248 <_main+0xb10>
100003ae8: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100003aec: ff c0                       	incl	%eax
100003aee: 39 c8                       	cmpl	%ecx, %eax
100003af0: 0f 82 d4 fe ff ff           	jb	-300 <_main+0xaea>
100003af6: c5 fb 10 15 82 35 00 00     	vmovsd	13698(%rip), %xmm2
100003afe: c5 eb 59 94 24 d8 00 00 00  	vmulsd	216(%rsp), %xmm2, %xmm2
100003b07: c5 f3 5c c0                 	vsubsd	%xmm0, %xmm1, %xmm0
100003b0b: c5 fb 58 05 75 35 00 00     	vaddsd	13685(%rip), %xmm0, %xmm0
100003b13: c5 fb 10 0d 75 35 00 00     	vmovsd	13685(%rip), %xmm1
100003b1b: c5 f3 5e c0                 	vdivsd	%xmm0, %xmm1, %xmm0
100003b1f: c5 eb 58 c0                 	vaddsd	%xmm0, %xmm2, %xmm0
100003b23: 8b 9c 24 28 02 00 00        	movl	552(%rsp), %ebx
100003b2a: c5 fb 11 84 24 d8 00 00 00  	vmovsd	%xmm0, 216(%rsp)
100003b33: c5 f8 77                    	vzeroupper
100003b36: e8 a5 33 00 00              	callq	13221 <dyld_stub_binder+0x100006ee0>
100003b3b: c5 fb 2c f0                 	vcvttsd2si	%xmm0, %esi
100003b3f: 4c 89 ef                    	movq	%r13, %rdi
100003b42: e8 63 33 00 00              	callq	13155 <dyld_stub_binder+0x100006eaa>
100003b47: 4c 89 ef                    	movq	%r13, %rdi
100003b4a: 31 f6                       	xorl	%esi, %esi
100003b4c: 48 8d 15 8a 53 00 00        	leaq	21386(%rip), %rdx
100003b53: e8 22 33 00 00              	callq	13090 <dyld_stub_binder+0x100006e7a>
100003b58: 48 8b 48 10                 	movq	16(%rax), %rcx
100003b5c: 48 89 8c 24 f0 00 00 00     	movq	%rcx, 240(%rsp)
100003b64: c5 f9 10 00                 	vmovupd	(%rax), %xmm0
100003b68: c5 f9 29 84 24 e0 00 00 00  	vmovapd	%xmm0, 224(%rsp)
100003b71: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003b75: c5 f9 11 00                 	vmovupd	%xmm0, (%rax)
100003b79: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003b81: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003b89: 48 8d 35 54 53 00 00        	leaq	21332(%rip), %rsi
100003b90: e8 d9 32 00 00              	callq	13017 <dyld_stub_binder+0x100006e6e>
100003b95: c4 e1 cb 2a c3              	vcvtsi2sd	%rbx, %xmm6, %xmm0
100003b9a: c5 fb 59 84 24 d8 00 00 00  	vmulsd	216(%rsp), %xmm0, %xmm0
100003ba3: c5 fb 5e 05 ed 34 00 00     	vdivsd	13549(%rip), %xmm0, %xmm0
100003bab: 48 8b 48 10                 	movq	16(%rax), %rcx
100003baf: 48 89 8c 24 d0 03 00 00     	movq	%rcx, 976(%rsp)
100003bb7: c5 f9 10 08                 	vmovupd	(%rax), %xmm1
100003bbb: c5 f9 29 8c 24 c0 03 00 00  	vmovapd	%xmm1, 960(%rsp)
100003bc4: c5 f1 57 c9                 	vxorpd	%xmm1, %xmm1, %xmm1
100003bc8: c5 f9 11 08                 	vmovupd	%xmm1, (%rax)
100003bcc: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003bd4: 48 8d bc 24 98 01 00 00     	leaq	408(%rsp), %rdi
100003bdc: e8 c3 32 00 00              	callq	12995 <dyld_stub_binder+0x100006ea4>
100003be1: 0f b6 94 24 98 01 00 00     	movzbl	408(%rsp), %edx
100003be9: f6 c2 01                    	testb	$1, %dl
100003bec: 48 8d 9c 24 d8 01 00 00     	leaq	472(%rsp), %rbx
100003bf4: 74 12                       	je	18 <_main+0xd28>
100003bf6: 48 8b b4 24 a8 01 00 00     	movq	424(%rsp), %rsi
100003bfe: 48 8b 94 24 a0 01 00 00     	movq	416(%rsp), %rdx
100003c06: eb 0b                       	jmp	11 <_main+0xd33>
100003c08: 48 d1 ea                    	shrq	%rdx
100003c0b: 48 8d b4 24 99 01 00 00     	leaq	409(%rsp), %rsi
100003c13: 4c 89 f7                    	movq	%r14, %rdi
100003c16: e8 59 32 00 00              	callq	12889 <dyld_stub_binder+0x100006e74>
100003c1b: 48 8b 48 10                 	movq	16(%rax), %rcx
100003c1f: 48 89 8c 24 70 01 00 00     	movq	%rcx, 368(%rsp)
100003c27: c5 f8 10 00                 	vmovups	(%rax), %xmm0
100003c2b: c5 f8 29 84 24 60 01 00 00  	vmovaps	%xmm0, 352(%rsp)
100003c34: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003c38: c5 f8 11 00                 	vmovups	%xmm0, (%rax)
100003c3c: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100003c44: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
100003c4c: 0f 85 7a 01 00 00           	jne	378 <_main+0xeec>
100003c52: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003c5a: 0f 85 87 01 00 00           	jne	391 <_main+0xf07>
100003c60: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
100003c68: 0f 85 94 01 00 00           	jne	404 <_main+0xf22>
100003c6e: 4d 89 ec                    	movq	%r13, %r12
100003c71: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003c79: 74 0d                       	je	13 <_main+0xda8>
100003c7b: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
100003c83: e8 2e 32 00 00              	callq	12846 <dyld_stub_binder+0x100006eb6>
100003c88: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
100003c94: c7 84 24 c0 03 00 00 00 00 01 03    	movl	$50397184, 960(%rsp)
100003c9f: 4c 8d 6c 24 78              	leaq	120(%rsp), %r13
100003ca4: 4c 89 ac 24 c8 03 00 00     	movq	%r13, 968(%rsp)
100003cac: 48 b8 1e 00 00 00 1e 00 00 00       	movabsq	$128849018910, %rax
100003cb6: 48 89 84 24 b8 01 00 00     	movq	%rax, 440(%rsp)
100003cbe: c5 fc 28 05 1a 34 00 00     	vmovaps	13338(%rip), %ymm0
100003cc6: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100003ccf: c7 44 24 08 00 00 00 00     	movl	$0, 8(%rsp)
100003cd7: c7 04 24 10 00 00 00        	movl	$16, (%rsp)
100003cde: 4c 89 f7                    	movq	%r14, %rdi
100003ce1: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003ce9: 48 8d 94 24 b8 01 00 00     	leaq	440(%rsp), %rdx
100003cf1: 31 c9                       	xorl	%ecx, %ecx
100003cf3: c5 fb 10 05 a5 33 00 00     	vmovsd	13221(%rip), %xmm0
100003cfb: 4c 8d 84 24 40 02 00 00     	leaq	576(%rsp), %r8
100003d03: 41 b9 02 00 00 00           	movl	$2, %r9d
100003d09: c5 f8 77                    	vzeroupper
100003d0c: e8 2d 31 00 00              	callq	12589 <dyld_stub_binder+0x100006e3e>
100003d11: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003d15: c5 f9 29 84 24 c0 03 00 00  	vmovapd	%xmm0, 960(%rsp)
100003d1e: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
100003d2a: c6 84 24 c0 03 00 00 0a     	movb	$10, 960(%rsp)
100003d32: 48 8d 84 24 c1 03 00 00     	leaq	961(%rsp), %rax
100003d3a: c6 40 04 65                 	movb	$101, 4(%rax)
100003d3e: c7 00 66 72 61 6d           	movl	$1835102822, (%rax)
100003d44: c6 84 24 c6 03 00 00 00     	movb	$0, 966(%rsp)
100003d4c: 48 c7 84 24 f0 00 00 00 00 00 00 00 	movq	$0, 240(%rsp)
100003d58: c7 84 24 e0 00 00 00 00 00 01 01    	movl	$16842752, 224(%rsp)
100003d63: 4c 89 ac 24 e8 00 00 00     	movq	%r13, 232(%rsp)
100003d6b: 4c 89 f7                    	movq	%r14, %rdi
100003d6e: 48 8d b4 24 e0 00 00 00     	leaq	224(%rsp), %rsi
100003d76: e8 b7 30 00 00              	callq	12471 <dyld_stub_binder+0x100006e32>
100003d7b: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003d83: 4d 89 e5                    	movq	%r12, %r13
100003d86: 4c 8d 74 24 68              	leaq	104(%rsp), %r14
100003d8b: 0f 85 94 00 00 00           	jne	148 <_main+0xf45>
100003d91: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003d99: 4c 8d 64 24 78              	leaq	120(%rsp), %r12
100003d9e: 0f 85 a1 00 00 00           	jne	161 <_main+0xf65>
100003da4: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
100003dac: 48 85 c0                    	testq	%rax, %rax
100003daf: 0f 84 ae 00 00 00           	je	174 <_main+0xf83>
100003db5: f0                          	lock
100003db6: ff 48 14                    	decl	20(%rax)
100003db9: 0f 85 a4 00 00 00           	jne	164 <_main+0xf83>
100003dbf: 4c 89 e7                    	movq	%r12, %rdi
100003dc2: e8 53 30 00 00              	callq	12371 <dyld_stub_binder+0x100006e1a>
100003dc7: e9 97 00 00 00              	jmp	151 <_main+0xf83>
100003dcc: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100003dd4: e8 dd 30 00 00              	callq	12509 <dyld_stub_binder+0x100006eb6>
100003dd9: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100003de1: 0f 84 79 fe ff ff           	je	-391 <_main+0xd80>
100003de7: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003def: e8 c2 30 00 00              	callq	12482 <dyld_stub_binder+0x100006eb6>
100003df4: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
100003dfc: 0f 84 6c fe ff ff           	je	-404 <_main+0xd8e>
100003e02: 48 8b bc 24 f0 00 00 00     	movq	240(%rsp), %rdi
100003e0a: e8 a7 30 00 00              	callq	12455 <dyld_stub_binder+0x100006eb6>
100003e0f: 4d 89 ec                    	movq	%r13, %r12
100003e12: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003e1a: 0f 85 5b fe ff ff           	jne	-421 <_main+0xd9b>
100003e20: e9 63 fe ff ff              	jmp	-413 <_main+0xda8>
100003e25: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003e2d: e8 84 30 00 00              	callq	12420 <dyld_stub_binder+0x100006eb6>
100003e32: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003e3a: 4c 8d 64 24 78              	leaq	120(%rsp), %r12
100003e3f: 0f 84 5f ff ff ff           	je	-161 <_main+0xec4>
100003e45: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
100003e4d: e8 64 30 00 00              	callq	12388 <dyld_stub_binder+0x100006eb6>
100003e52: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
100003e5a: 48 85 c0                    	testq	%rax, %rax
100003e5d: 0f 85 52 ff ff ff           	jne	-174 <_main+0xed5>
100003e63: 48 c7 84 24 b0 00 00 00 00 00 00 00 	movq	$0, 176(%rsp)
100003e6f: 48 8d 44 24 7c              	leaq	124(%rsp), %rax
100003e74: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003e78: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100003e7d: 83 7c 24 7c 00              	cmpl	$0, 124(%rsp)
100003e82: 7e 20                       	jle	32 <_main+0xfc4>
100003e84: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003e8c: 31 c9                       	xorl	%ecx, %ecx
100003e8e: 66 90                       	nop
100003e90: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003e97: 48 ff c1                    	incq	%rcx
100003e9a: 48 63 54 24 7c              	movslq	124(%rsp), %rdx
100003e9f: 48 39 d1                    	cmpq	%rdx, %rcx
100003ea2: 7c ec                       	jl	-20 <_main+0xfb0>
100003ea4: 48 8b bc 24 c0 00 00 00     	movq	192(%rsp), %rdi
100003eac: 48 8d 84 24 c8 00 00 00     	leaq	200(%rsp), %rax
100003eb4: 48 39 c7                    	cmpq	%rax, %rdi
100003eb7: 74 08                       	je	8 <_main+0xfe1>
100003eb9: c5 f8 77                    	vzeroupper
100003ebc: e8 8f 2f 00 00              	callq	12175 <dyld_stub_binder+0x100006e50>
100003ec1: 48 8b 84 24 38 01 00 00     	movq	312(%rsp), %rax
100003ec9: 48 85 c0                    	testq	%rax, %rax
100003ecc: 74 16                       	je	22 <_main+0x1004>
100003ece: f0                          	lock
100003ecf: ff 48 14                    	decl	20(%rax)
100003ed2: 75 10                       	jne	16 <_main+0x1004>
100003ed4: 48 8d bc 24 00 01 00 00     	leaq	256(%rsp), %rdi
100003edc: c5 f8 77                    	vzeroupper
100003edf: e8 36 2f 00 00              	callq	12086 <dyld_stub_binder+0x100006e1a>
100003ee4: 48 c7 84 24 38 01 00 00 00 00 00 00 	movq	$0, 312(%rsp)
100003ef0: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
100003ef8: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100003efc: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
100003f00: 83 bc 24 04 01 00 00 00     	cmpl	$0, 260(%rsp)
100003f08: 7e 2d                       	jle	45 <_main+0x1057>
100003f0a: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
100003f12: 31 c9                       	xorl	%ecx, %ecx
100003f14: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003f1e: 66 90                       	nop
100003f20: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003f27: 48 ff c1                    	incq	%rcx
100003f2a: 48 63 94 24 04 01 00 00     	movslq	260(%rsp), %rdx
100003f32: 48 39 d1                    	cmpq	%rdx, %rcx
100003f35: 7c e9                       	jl	-23 <_main+0x1040>
100003f37: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
100003f3f: 48 8d 84 24 50 01 00 00     	leaq	336(%rsp), %rax
100003f47: 48 39 c7                    	cmpq	%rax, %rdi
100003f4a: 74 08                       	je	8 <_main+0x1074>
100003f4c: c5 f8 77                    	vzeroupper
100003f4f: e8 fc 2e 00 00              	callq	12028 <dyld_stub_binder+0x100006e50>
100003f54: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100003f5c: c5 f8 77                    	vzeroupper
100003f5f: e8 9c 04 00 00              	callq	1180 <_main+0x1520>
100003f64: 45 31 e4                    	xorl	%r12d, %r12d
100003f67: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003f6c: 48 85 c0                    	testq	%rax, %rax
100003f6f: 0f 85 55 f9 ff ff           	jne	-1707 <_main+0x9ea>
100003f75: e9 63 f9 ff ff              	jmp	-1693 <_main+0x9fd>
100003f7a: 48 8b 3d c7 50 00 00        	movq	20679(%rip), %rdi
100003f81: 48 8d 35 70 4f 00 00        	leaq	20336(%rip), %rsi
100003f88: ba 0d 00 00 00              	movl	$13, %edx
100003f8d: c5 f8 77                    	vzeroupper
100003f90: e8 fb 05 00 00              	callq	1531 <_main+0x16b0>
100003f95: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100003f9d: e8 6c 2e 00 00              	callq	11884 <dyld_stub_binder+0x100006e0e>
100003fa2: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
100003faa: e8 11 0a 00 00              	callq	2577 <__ZN14ModelInterfaceD2Ev>
100003faf: 48 8b 05 aa 50 00 00        	movq	20650(%rip), %rax
100003fb6: 48 8b 00                    	movq	(%rax), %rax
100003fb9: 48 3b 84 24 e0 03 00 00     	cmpq	992(%rsp), %rax
100003fc1: 75 11                       	jne	17 <_main+0x10f4>
100003fc3: 31 c0                       	xorl	%eax, %eax
100003fc5: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100003fc9: 5b                          	popq	%rbx
100003fca: 41 5c                       	popq	%r12
100003fcc: 41 5d                       	popq	%r13
100003fce: 41 5e                       	popq	%r14
100003fd0: 41 5f                       	popq	%r15
100003fd2: 5d                          	popq	%rbp
100003fd3: c3                          	retq
100003fd4: e8 fb 2e 00 00              	callq	12027 <dyld_stub_binder+0x100006ed4>
100003fd9: e9 e7 03 00 00              	jmp	999 <_main+0x14e5>
100003fde: 48 89 c3                    	movq	%rax, %rbx
100003fe1: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
100003fe9: 0f 84 e9 03 00 00           	je	1001 <_main+0x14f8>
100003fef: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
100003ff7: e8 ba 2e 00 00              	callq	11962 <dyld_stub_binder+0x100006eb6>
100003ffc: e9 d7 03 00 00              	jmp	983 <_main+0x14f8>
100004001: 48 89 c3                    	movq	%rax, %rbx
100004004: e9 cf 03 00 00              	jmp	975 <_main+0x14f8>
100004009: 48 89 c7                    	movq	%rax, %rdi
10000400c: e8 df 03 00 00              	callq	991 <_main+0x1510>
100004011: 48 89 c7                    	movq	%rax, %rdi
100004014: e8 d7 03 00 00              	callq	983 <_main+0x1510>
100004019: 48 89 c7                    	movq	%rax, %rdi
10000401c: e8 cf 03 00 00              	callq	975 <_main+0x1510>
100004021: 48 89 c3                    	movq	%rax, %rbx
100004024: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
10000402c: 48 85 c0                    	testq	%rax, %rax
10000402f: 0f 85 c8 01 00 00           	jne	456 <_main+0x131d>
100004035: e9 d3 01 00 00              	jmp	467 <_main+0x132d>
10000403a: 48 89 c3                    	movq	%rax, %rbx
10000403d: 48 8b 84 24 38 01 00 00     	movq	312(%rsp), %rax
100004045: 48 85 c0                    	testq	%rax, %rax
100004048: 74 13                       	je	19 <_main+0x117d>
10000404a: f0                          	lock
10000404b: ff 48 14                    	decl	20(%rax)
10000404e: 75 0d                       	jne	13 <_main+0x117d>
100004050: 48 8d bc 24 00 01 00 00     	leaq	256(%rsp), %rdi
100004058: e8 bd 2d 00 00              	callq	11709 <dyld_stub_binder+0x100006e1a>
10000405d: 48 c7 84 24 38 01 00 00 00 00 00 00 	movq	$0, 312(%rsp)
100004069: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000406d: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
100004075: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100004079: 83 bc 24 04 01 00 00 00     	cmpl	$0, 260(%rsp)
100004081: 7e 21                       	jle	33 <_main+0x11c4>
100004083: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
10000408b: 31 c9                       	xorl	%ecx, %ecx
10000408d: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004094: 48 ff c1                    	incq	%rcx
100004097: 48 63 94 24 04 01 00 00     	movslq	260(%rsp), %rdx
10000409f: 48 39 d1                    	cmpq	%rdx, %rcx
1000040a2: 7c e9                       	jl	-23 <_main+0x11ad>
1000040a4: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
1000040ac: 48 8d 84 24 50 01 00 00     	leaq	336(%rsp), %rax
1000040b4: 48 39 c7                    	cmpq	%rax, %rdi
1000040b7: 0f 84 88 02 00 00           	je	648 <_main+0x1465>
1000040bd: c5 f8 77                    	vzeroupper
1000040c0: e8 8b 2d 00 00              	callq	11659 <dyld_stub_binder+0x100006e50>
1000040c5: e9 7b 02 00 00              	jmp	635 <_main+0x1465>
1000040ca: 48 89 c7                    	movq	%rax, %rdi
1000040cd: e8 1e 03 00 00              	callq	798 <_main+0x1510>
1000040d2: 48 89 c3                    	movq	%rax, %rbx
1000040d5: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000040da: 48 85 c0                    	testq	%rax, %rax
1000040dd: 0f 85 6c 02 00 00           	jne	620 <_main+0x146f>
1000040e3: e9 7a 02 00 00              	jmp	634 <_main+0x1482>
1000040e8: 48 89 c3                    	movq	%rax, %rbx
1000040eb: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000040f3: 74 1f                       	je	31 <_main+0x1234>
1000040f5: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
1000040fd: e8 b4 2d 00 00              	callq	11700 <dyld_stub_binder+0x100006eb6>
100004102: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
10000410a: 75 16                       	jne	22 <_main+0x1242>
10000410c: e9 df 00 00 00              	jmp	223 <_main+0x1310>
100004111: 48 89 c3                    	movq	%rax, %rbx
100004114: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
10000411c: 0f 84 ce 00 00 00           	je	206 <_main+0x1310>
100004122: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
10000412a: e9 aa 00 00 00              	jmp	170 <_main+0x12f9>
10000412f: 48 89 c3                    	movq	%rax, %rbx
100004132: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
10000413a: 75 23                       	jne	35 <_main+0x127f>
10000413c: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100004144: 75 3f                       	jne	63 <_main+0x12a5>
100004146: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
10000414e: 75 5b                       	jne	91 <_main+0x12cb>
100004150: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100004158: 75 77                       	jne	119 <_main+0x12f1>
10000415a: e9 91 00 00 00              	jmp	145 <_main+0x1310>
10000415f: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100004167: e8 4a 2d 00 00              	callq	11594 <dyld_stub_binder+0x100006eb6>
10000416c: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100004174: 74 d0                       	je	-48 <_main+0x1266>
100004176: eb 0d                       	jmp	13 <_main+0x12a5>
100004178: 48 89 c3                    	movq	%rax, %rbx
10000417b: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
100004183: 74 c1                       	je	-63 <_main+0x1266>
100004185: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
10000418d: e8 24 2d 00 00              	callq	11556 <dyld_stub_binder+0x100006eb6>
100004192: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
10000419a: 74 b4                       	je	-76 <_main+0x1270>
10000419c: eb 0d                       	jmp	13 <_main+0x12cb>
10000419e: 48 89 c3                    	movq	%rax, %rbx
1000041a1: f6 84 24 e0 00 00 00 01     	testb	$1, 224(%rsp)
1000041a9: 74 a5                       	je	-91 <_main+0x1270>
1000041ab: 48 8b bc 24 f0 00 00 00     	movq	240(%rsp), %rdi
1000041b3: e8 fe 2c 00 00              	callq	11518 <dyld_stub_binder+0x100006eb6>
1000041b8: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
1000041c0: 75 0f                       	jne	15 <_main+0x12f1>
1000041c2: eb 2c                       	jmp	44 <_main+0x1310>
1000041c4: 48 89 c3                    	movq	%rax, %rbx
1000041c7: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
1000041cf: 74 1f                       	je	31 <_main+0x1310>
1000041d1: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
1000041d9: e8 d8 2c 00 00              	callq	11480 <dyld_stub_binder+0x100006eb6>
1000041de: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
1000041e6: 48 85 c0                    	testq	%rax, %rax
1000041e9: 75 12                       	jne	18 <_main+0x131d>
1000041eb: eb 20                       	jmp	32 <_main+0x132d>
1000041ed: 48 89 c3                    	movq	%rax, %rbx
1000041f0: 48 8b 84 24 b0 00 00 00     	movq	176(%rsp), %rax
1000041f8: 48 85 c0                    	testq	%rax, %rax
1000041fb: 74 10                       	je	16 <_main+0x132d>
1000041fd: f0                          	lock
1000041fe: ff 48 14                    	decl	20(%rax)
100004201: 75 0a                       	jne	10 <_main+0x132d>
100004203: 48 8d 7c 24 78              	leaq	120(%rsp), %rdi
100004208: e8 0d 2c 00 00              	callq	11277 <dyld_stub_binder+0x100006e1a>
10000420d: 48 c7 84 24 b0 00 00 00 00 00 00 00 	movq	$0, 176(%rsp)
100004219: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
10000421d: 48 8d 44 24 7c              	leaq	124(%rsp), %rax
100004222: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100004227: 83 7c 24 7c 00              	cmpl	$0, 124(%rsp)
10000422c: 7e 1e                       	jle	30 <_main+0x136c>
10000422e: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100004236: 31 c9                       	xorl	%ecx, %ecx
100004238: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
10000423f: 48 ff c1                    	incq	%rcx
100004242: 48 63 54 24 7c              	movslq	124(%rsp), %rdx
100004247: 48 39 d1                    	cmpq	%rdx, %rcx
10000424a: 7c ec                       	jl	-20 <_main+0x1358>
10000424c: 48 8b bc 24 c0 00 00 00     	movq	192(%rsp), %rdi
100004254: 48 8d 84 24 c8 00 00 00     	leaq	200(%rsp), %rax
10000425c: 48 39 c7                    	cmpq	%rax, %rdi
10000425f: 74 1f                       	je	31 <_main+0x13a0>
100004261: c5 f8 77                    	vzeroupper
100004264: e8 e7 2b 00 00              	callq	11239 <dyld_stub_binder+0x100006e50>
100004269: eb 15                       	jmp	21 <_main+0x13a0>
10000426b: 48 89 c7                    	movq	%rax, %rdi
10000426e: e8 7d 01 00 00              	callq	381 <_main+0x1510>
100004273: eb 08                       	jmp	8 <_main+0x139d>
100004275: 48 89 c3                    	movq	%rax, %rbx
100004278: e9 8a 00 00 00              	jmp	138 <_main+0x1427>
10000427d: 48 89 c3                    	movq	%rax, %rbx
100004280: 48 8b 84 24 38 01 00 00     	movq	312(%rsp), %rax
100004288: 48 85 c0                    	testq	%rax, %rax
10000428b: 74 16                       	je	22 <_main+0x13c3>
10000428d: f0                          	lock
10000428e: ff 48 14                    	decl	20(%rax)
100004291: 75 10                       	jne	16 <_main+0x13c3>
100004293: 48 8d bc 24 00 01 00 00     	leaq	256(%rsp), %rdi
10000429b: c5 f8 77                    	vzeroupper
10000429e: e8 77 2b 00 00              	callq	11127 <dyld_stub_binder+0x100006e1a>
1000042a3: 48 c7 84 24 38 01 00 00 00 00 00 00 	movq	$0, 312(%rsp)
1000042af: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
1000042b3: 48 8d 84 24 10 01 00 00     	leaq	272(%rsp), %rax
1000042bb: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
1000042bf: 83 bc 24 04 01 00 00 00     	cmpl	$0, 260(%rsp)
1000042c7: 7e 21                       	jle	33 <_main+0x140a>
1000042c9: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
1000042d1: 31 c9                       	xorl	%ecx, %ecx
1000042d3: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000042da: 48 ff c1                    	incq	%rcx
1000042dd: 48 63 94 24 04 01 00 00     	movslq	260(%rsp), %rdx
1000042e5: 48 39 d1                    	cmpq	%rdx, %rcx
1000042e8: 7c e9                       	jl	-23 <_main+0x13f3>
1000042ea: 48 8b bc 24 48 01 00 00     	movq	328(%rsp), %rdi
1000042f2: 48 8d 84 24 50 01 00 00     	leaq	336(%rsp), %rax
1000042fa: 48 39 c7                    	cmpq	%rax, %rdi
1000042fd: 74 08                       	je	8 <_main+0x1427>
1000042ff: c5 f8 77                    	vzeroupper
100004302: e8 49 2b 00 00              	callq	11081 <dyld_stub_binder+0x100006e50>
100004307: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
10000430f: c5 f8 77                    	vzeroupper
100004312: e8 e9 00 00 00              	callq	233 <_main+0x1520>
100004317: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000431c: 48 85 c0                    	testq	%rax, %rax
10000431f: 75 2e                       	jne	46 <_main+0x146f>
100004321: eb 3f                       	jmp	63 <_main+0x1482>
100004323: 48 89 c7                    	movq	%rax, %rdi
100004326: e8 c5 00 00 00              	callq	197 <_main+0x1510>
10000432b: 48 89 c3                    	movq	%rax, %rbx
10000432e: 48 8b 44 24 50              	movq	80(%rsp), %rax
100004333: 48 85 c0                    	testq	%rax, %rax
100004336: 75 17                       	jne	23 <_main+0x146f>
100004338: eb 28                       	jmp	40 <_main+0x1482>
10000433a: 48 89 c7                    	movq	%rax, %rdi
10000433d: e8 ae 00 00 00              	callq	174 <_main+0x1510>
100004342: 48 89 c3                    	movq	%rax, %rbx
100004345: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000434a: 48 85 c0                    	testq	%rax, %rax
10000434d: 74 13                       	je	19 <_main+0x1482>
10000434f: f0                          	lock
100004350: ff 48 14                    	decl	20(%rax)
100004353: 75 0d                       	jne	13 <_main+0x1482>
100004355: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
10000435a: c5 f8 77                    	vzeroupper
10000435d: e8 b8 2a 00 00              	callq	10936 <dyld_stub_binder+0x100006e1a>
100004362: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
10000436b: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000436f: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
100004374: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100004379: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
10000437e: 7e 24                       	jle	36 <_main+0x14c4>
100004380: 48 8b 44 24 58              	movq	88(%rsp), %rax
100004385: 31 c9                       	xorl	%ecx, %ecx
100004387: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100004390: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004397: 48 ff c1                    	incq	%rcx
10000439a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000439f: 48 39 d1                    	cmpq	%rdx, %rcx
1000043a2: 7c ec                       	jl	-20 <_main+0x14b0>
1000043a4: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
1000043a9: 48 8d 44 24 68              	leaq	104(%rsp), %rax
1000043ae: 48 39 c7                    	cmpq	%rax, %rdi
1000043b1: 74 15                       	je	21 <_main+0x14e8>
1000043b3: c5 f8 77                    	vzeroupper
1000043b6: e8 95 2a 00 00              	callq	10901 <dyld_stub_binder+0x100006e50>
1000043bb: eb 0b                       	jmp	11 <_main+0x14e8>
1000043bd: 48 89 c7                    	movq	%rax, %rdi
1000043c0: e8 2b 00 00 00              	callq	43 <_main+0x1510>
1000043c5: 48 89 c3                    	movq	%rax, %rbx
1000043c8: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
1000043d0: c5 f8 77                    	vzeroupper
1000043d3: e8 36 2a 00 00              	callq	10806 <dyld_stub_binder+0x100006e0e>
1000043d8: 48 8d bc 24 08 02 00 00     	leaq	520(%rsp), %rdi
1000043e0: e8 db 05 00 00              	callq	1499 <__ZN14ModelInterfaceD2Ev>
1000043e5: 48 89 df                    	movq	%rbx, %rdi
1000043e8: e8 15 2a 00 00              	callq	10773 <dyld_stub_binder+0x100006e02>
1000043ed: 0f 0b                       	ud2
1000043ef: 90                          	nop
1000043f0: 50                          	pushq	%rax
1000043f1: e8 d2 2a 00 00              	callq	10962 <dyld_stub_binder+0x100006ec8>
1000043f6: e8 b5 2a 00 00              	callq	10933 <dyld_stub_binder+0x100006eb0>
1000043fb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004400: 55                          	pushq	%rbp
100004401: 48 89 e5                    	movq	%rsp, %rbp
100004404: 53                          	pushq	%rbx
100004405: 50                          	pushq	%rax
100004406: 48 89 fb                    	movq	%rdi, %rbx
100004409: 48 8b 87 08 01 00 00        	movq	264(%rdi), %rax
100004410: 48 85 c0                    	testq	%rax, %rax
100004413: 74 12                       	je	18 <_main+0x1547>
100004415: f0                          	lock
100004416: ff 48 14                    	decl	20(%rax)
100004419: 75 0c                       	jne	12 <_main+0x1547>
10000441b: 48 8d bb d0 00 00 00        	leaq	208(%rbx), %rdi
100004422: e8 f3 29 00 00              	callq	10739 <dyld_stub_binder+0x100006e1a>
100004427: 48 c7 83 08 01 00 00 00 00 00 00    	movq	$0, 264(%rbx)
100004432: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004436: c5 fc 11 83 e0 00 00 00     	vmovups	%ymm0, 224(%rbx)
10000443e: 83 bb d4 00 00 00 00        	cmpl	$0, 212(%rbx)
100004445: 7e 1f                       	jle	31 <_main+0x1586>
100004447: 48 8b 83 10 01 00 00        	movq	272(%rbx), %rax
10000444e: 31 c9                       	xorl	%ecx, %ecx
100004450: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004457: 48 ff c1                    	incq	%rcx
10000445a: 48 63 93 d4 00 00 00        	movslq	212(%rbx), %rdx
100004461: 48 39 d1                    	cmpq	%rdx, %rcx
100004464: 7c ea                       	jl	-22 <_main+0x1570>
100004466: 48 8b bb 18 01 00 00        	movq	280(%rbx), %rdi
10000446d: 48 8d 83 20 01 00 00        	leaq	288(%rbx), %rax
100004474: 48 39 c7                    	cmpq	%rax, %rdi
100004477: 74 08                       	je	8 <_main+0x15a1>
100004479: c5 f8 77                    	vzeroupper
10000447c: e8 cf 29 00 00              	callq	10703 <dyld_stub_binder+0x100006e50>
100004481: 48 8b 83 a8 00 00 00        	movq	168(%rbx), %rax
100004488: 48 85 c0                    	testq	%rax, %rax
10000448b: 74 12                       	je	18 <_main+0x15bf>
10000448d: f0                          	lock
10000448e: ff 48 14                    	decl	20(%rax)
100004491: 75 0c                       	jne	12 <_main+0x15bf>
100004493: 48 8d 7b 70                 	leaq	112(%rbx), %rdi
100004497: c5 f8 77                    	vzeroupper
10000449a: e8 7b 29 00 00              	callq	10619 <dyld_stub_binder+0x100006e1a>
10000449f: 48 c7 83 a8 00 00 00 00 00 00 00    	movq	$0, 168(%rbx)
1000044aa: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000044ae: c5 fc 11 83 80 00 00 00     	vmovups	%ymm0, 128(%rbx)
1000044b6: 83 7b 74 00                 	cmpl	$0, 116(%rbx)
1000044ba: 7e 27                       	jle	39 <_main+0x1603>
1000044bc: 48 8b 83 b0 00 00 00        	movq	176(%rbx), %rax
1000044c3: 31 c9                       	xorl	%ecx, %ecx
1000044c5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000044cf: 90                          	nop
1000044d0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000044d7: 48 ff c1                    	incq	%rcx
1000044da: 48 63 53 74                 	movslq	116(%rbx), %rdx
1000044de: 48 39 d1                    	cmpq	%rdx, %rcx
1000044e1: 7c ed                       	jl	-19 <_main+0x15f0>
1000044e3: 48 8b bb b8 00 00 00        	movq	184(%rbx), %rdi
1000044ea: 48 8d 83 c0 00 00 00        	leaq	192(%rbx), %rax
1000044f1: 48 39 c7                    	cmpq	%rax, %rdi
1000044f4: 74 08                       	je	8 <_main+0x161e>
1000044f6: c5 f8 77                    	vzeroupper
1000044f9: e8 52 29 00 00              	callq	10578 <dyld_stub_binder+0x100006e50>
1000044fe: 48 8b 43 48                 	movq	72(%rbx), %rax
100004502: 48 85 c0                    	testq	%rax, %rax
100004505: 74 12                       	je	18 <_main+0x1639>
100004507: f0                          	lock
100004508: ff 48 14                    	decl	20(%rax)
10000450b: 75 0c                       	jne	12 <_main+0x1639>
10000450d: 48 8d 7b 10                 	leaq	16(%rbx), %rdi
100004511: c5 f8 77                    	vzeroupper
100004514: e8 01 29 00 00              	callq	10497 <dyld_stub_binder+0x100006e1a>
100004519: 48 c7 43 48 00 00 00 00     	movq	$0, 72(%rbx)
100004521: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004525: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
10000452a: 83 7b 14 00                 	cmpl	$0, 20(%rbx)
10000452e: 7e 23                       	jle	35 <_main+0x1673>
100004530: 48 8b 43 50                 	movq	80(%rbx), %rax
100004534: 31 c9                       	xorl	%ecx, %ecx
100004536: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004540: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100004547: 48 ff c1                    	incq	%rcx
10000454a: 48 63 53 14                 	movslq	20(%rbx), %rdx
10000454e: 48 39 d1                    	cmpq	%rdx, %rcx
100004551: 7c ed                       	jl	-19 <_main+0x1660>
100004553: 48 8b 7b 58                 	movq	88(%rbx), %rdi
100004557: 48 83 c3 60                 	addq	$96, %rbx
10000455b: 48 39 df                    	cmpq	%rbx, %rdi
10000455e: 74 08                       	je	8 <_main+0x1688>
100004560: c5 f8 77                    	vzeroupper
100004563: e8 e8 28 00 00              	callq	10472 <dyld_stub_binder+0x100006e50>
100004568: 48 83 c4 08                 	addq	$8, %rsp
10000456c: 5b                          	popq	%rbx
10000456d: 5d                          	popq	%rbp
10000456e: c5 f8 77                    	vzeroupper
100004571: c3                          	retq
100004572: 48 89 c7                    	movq	%rax, %rdi
100004575: e8 76 fe ff ff              	callq	-394 <_main+0x1510>
10000457a: 48 89 c7                    	movq	%rax, %rdi
10000457d: e8 6e fe ff ff              	callq	-402 <_main+0x1510>
100004582: 48 89 c7                    	movq	%rax, %rdi
100004585: e8 66 fe ff ff              	callq	-410 <_main+0x1510>
10000458a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100004590: 55                          	pushq	%rbp
100004591: 48 89 e5                    	movq	%rsp, %rbp
100004594: 41 57                       	pushq	%r15
100004596: 41 56                       	pushq	%r14
100004598: 41 55                       	pushq	%r13
10000459a: 41 54                       	pushq	%r12
10000459c: 53                          	pushq	%rbx
10000459d: 48 83 ec 28                 	subq	$40, %rsp
1000045a1: 49 89 d6                    	movq	%rdx, %r14
1000045a4: 49 89 f7                    	movq	%rsi, %r15
1000045a7: 48 89 fb                    	movq	%rdi, %rbx
1000045aa: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
1000045ae: 48 89 de                    	movq	%rbx, %rsi
1000045b1: e8 ca 28 00 00              	callq	10442 <dyld_stub_binder+0x100006e80>
1000045b6: 80 7d b0 00                 	cmpb	$0, -80(%rbp)
1000045ba: 0f 84 ae 00 00 00           	je	174 <_main+0x178e>
1000045c0: 48 8b 03                    	movq	(%rbx), %rax
1000045c3: 48 8b 40 e8                 	movq	-24(%rax), %rax
1000045c7: 4c 8d 24 03                 	leaq	(%rbx,%rax), %r12
1000045cb: 48 8b 7c 03 28              	movq	40(%rbx,%rax), %rdi
1000045d0: 44 8b 6c 03 08              	movl	8(%rbx,%rax), %r13d
1000045d5: 8b 84 03 90 00 00 00        	movl	144(%rbx,%rax), %eax
1000045dc: 83 f8 ff                    	cmpl	$-1, %eax
1000045df: 75 4a                       	jne	74 <_main+0x174b>
1000045e1: 48 89 7d c0                 	movq	%rdi, -64(%rbp)
1000045e5: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
1000045e9: 4c 89 e6                    	movq	%r12, %rsi
1000045ec: e8 77 28 00 00              	callq	10359 <dyld_stub_binder+0x100006e68>
1000045f1: 48 8b 35 58 4a 00 00        	movq	19032(%rip), %rsi
1000045f8: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
1000045fc: e8 61 28 00 00              	callq	10337 <dyld_stub_binder+0x100006e62>
100004601: 48 8b 08                    	movq	(%rax), %rcx
100004604: 48 89 c7                    	movq	%rax, %rdi
100004607: be 20 00 00 00              	movl	$32, %esi
10000460c: ff 51 38                    	callq	*56(%rcx)
10000460f: 88 45 d7                    	movb	%al, -41(%rbp)
100004612: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004616: e8 77 28 00 00              	callq	10359 <dyld_stub_binder+0x100006e92>
10000461b: 0f be 45 d7                 	movsbl	-41(%rbp), %eax
10000461f: 41 89 84 24 90 00 00 00     	movl	%eax, 144(%r12)
100004627: 48 8b 7d c0                 	movq	-64(%rbp), %rdi
10000462b: 4d 01 fe                    	addq	%r15, %r14
10000462e: 41 81 e5 b0 00 00 00        	andl	$176, %r13d
100004635: 41 83 fd 20                 	cmpl	$32, %r13d
100004639: 4c 89 fa                    	movq	%r15, %rdx
10000463c: 49 0f 44 d6                 	cmoveq	%r14, %rdx
100004640: 44 0f be c8                 	movsbl	%al, %r9d
100004644: 4c 89 fe                    	movq	%r15, %rsi
100004647: 4c 89 f1                    	movq	%r14, %rcx
10000464a: 4d 89 e0                    	movq	%r12, %r8
10000464d: e8 9e 00 00 00              	callq	158 <_main+0x1810>
100004652: 48 85 c0                    	testq	%rax, %rax
100004655: 75 17                       	jne	23 <_main+0x178e>
100004657: 48 8b 03                    	movq	(%rbx), %rax
10000465a: 48 8b 40 e8                 	movq	-24(%rax), %rax
10000465e: 48 8d 3c 03                 	leaq	(%rbx,%rax), %rdi
100004662: 8b 74 03 20                 	movl	32(%rbx,%rax), %esi
100004666: 83 ce 05                    	orl	$5, %esi
100004669: e8 30 28 00 00              	callq	10288 <dyld_stub_binder+0x100006e9e>
10000466e: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100004672: e8 0f 28 00 00              	callq	10255 <dyld_stub_binder+0x100006e86>
100004677: 48 89 d8                    	movq	%rbx, %rax
10000467a: 48 83 c4 28                 	addq	$40, %rsp
10000467e: 5b                          	popq	%rbx
10000467f: 41 5c                       	popq	%r12
100004681: 41 5d                       	popq	%r13
100004683: 41 5e                       	popq	%r14
100004685: 41 5f                       	popq	%r15
100004687: 5d                          	popq	%rbp
100004688: c3                          	retq
100004689: eb 0e                       	jmp	14 <_main+0x17b9>
10000468b: 49 89 c6                    	movq	%rax, %r14
10000468e: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100004692: e8 fb 27 00 00              	callq	10235 <dyld_stub_binder+0x100006e92>
100004697: eb 03                       	jmp	3 <_main+0x17bc>
100004699: 49 89 c6                    	movq	%rax, %r14
10000469c: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
1000046a0: e8 e1 27 00 00              	callq	10209 <dyld_stub_binder+0x100006e86>
1000046a5: eb 03                       	jmp	3 <_main+0x17ca>
1000046a7: 49 89 c6                    	movq	%rax, %r14
1000046aa: 4c 89 f7                    	movq	%r14, %rdi
1000046ad: e8 16 28 00 00              	callq	10262 <dyld_stub_binder+0x100006ec8>
1000046b2: 48 8b 03                    	movq	(%rbx), %rax
1000046b5: 48 8b 78 e8                 	movq	-24(%rax), %rdi
1000046b9: 48 01 df                    	addq	%rbx, %rdi
1000046bc: e8 d7 27 00 00              	callq	10199 <dyld_stub_binder+0x100006e98>
1000046c1: e8 08 28 00 00              	callq	10248 <dyld_stub_binder+0x100006ece>
1000046c6: eb af                       	jmp	-81 <_main+0x1797>
1000046c8: 48 89 c3                    	movq	%rax, %rbx
1000046cb: e8 fe 27 00 00              	callq	10238 <dyld_stub_binder+0x100006ece>
1000046d0: 48 89 df                    	movq	%rbx, %rdi
1000046d3: e8 2a 27 00 00              	callq	10026 <dyld_stub_binder+0x100006e02>
1000046d8: 0f 0b                       	ud2
1000046da: 48 89 c7                    	movq	%rax, %rdi
1000046dd: e8 0e fd ff ff              	callq	-754 <_main+0x1510>
1000046e2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000046ec: 0f 1f 40 00                 	nopl	(%rax)
1000046f0: 55                          	pushq	%rbp
1000046f1: 48 89 e5                    	movq	%rsp, %rbp
1000046f4: 41 57                       	pushq	%r15
1000046f6: 41 56                       	pushq	%r14
1000046f8: 41 55                       	pushq	%r13
1000046fa: 41 54                       	pushq	%r12
1000046fc: 53                          	pushq	%rbx
1000046fd: 48 83 ec 38                 	subq	$56, %rsp
100004701: 48 85 ff                    	testq	%rdi, %rdi
100004704: 0f 84 17 01 00 00           	je	279 <_main+0x1941>
10000470a: 4d 89 c4                    	movq	%r8, %r12
10000470d: 49 89 cf                    	movq	%rcx, %r15
100004710: 49 89 fe                    	movq	%rdi, %r14
100004713: 44 89 4d bc                 	movl	%r9d, -68(%rbp)
100004717: 48 89 c8                    	movq	%rcx, %rax
10000471a: 48 29 f0                    	subq	%rsi, %rax
10000471d: 49 8b 48 18                 	movq	24(%r8), %rcx
100004721: 45 31 ed                    	xorl	%r13d, %r13d
100004724: 48 29 c1                    	subq	%rax, %rcx
100004727: 4c 0f 4f e9                 	cmovgq	%rcx, %r13
10000472b: 48 89 55 a8                 	movq	%rdx, -88(%rbp)
10000472f: 48 89 d3                    	movq	%rdx, %rbx
100004732: 48 29 f3                    	subq	%rsi, %rbx
100004735: 48 85 db                    	testq	%rbx, %rbx
100004738: 7e 15                       	jle	21 <_main+0x186f>
10000473a: 49 8b 06                    	movq	(%r14), %rax
10000473d: 4c 89 f7                    	movq	%r14, %rdi
100004740: 48 89 da                    	movq	%rbx, %rdx
100004743: ff 50 60                    	callq	*96(%rax)
100004746: 48 39 d8                    	cmpq	%rbx, %rax
100004749: 0f 85 d2 00 00 00           	jne	210 <_main+0x1941>
10000474f: 4d 85 ed                    	testq	%r13, %r13
100004752: 0f 8e a1 00 00 00           	jle	161 <_main+0x1919>
100004758: 4c 89 65 b0                 	movq	%r12, -80(%rbp)
10000475c: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004760: c5 f8 29 45 c0              	vmovaps	%xmm0, -64(%rbp)
100004765: 48 c7 45 d0 00 00 00 00     	movq	$0, -48(%rbp)
10000476d: 49 83 fd 17                 	cmpq	$23, %r13
100004771: 73 12                       	jae	18 <_main+0x18a5>
100004773: 43 8d 44 2d 00              	leal	(%r13,%r13), %eax
100004778: 88 45 c0                    	movb	%al, -64(%rbp)
10000477b: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
10000477f: 4c 8d 65 c1                 	leaq	-63(%rbp), %r12
100004783: eb 27                       	jmp	39 <_main+0x18cc>
100004785: 49 8d 5d 10                 	leaq	16(%r13), %rbx
100004789: 48 83 e3 f0                 	andq	$-16, %rbx
10000478d: 48 89 df                    	movq	%rbx, %rdi
100004790: e8 2d 27 00 00              	callq	10029 <dyld_stub_binder+0x100006ec2>
100004795: 49 89 c4                    	movq	%rax, %r12
100004798: 48 89 45 d0                 	movq	%rax, -48(%rbp)
10000479c: 48 83 cb 01                 	orq	$1, %rbx
1000047a0: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
1000047a4: 4c 89 6d c8                 	movq	%r13, -56(%rbp)
1000047a8: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
1000047ac: 0f b6 75 bc                 	movzbl	-68(%rbp), %esi
1000047b0: 4c 89 e7                    	movq	%r12, %rdi
1000047b3: 4c 89 ea                    	movq	%r13, %rdx
1000047b6: e8 1f 27 00 00              	callq	10015 <dyld_stub_binder+0x100006eda>
1000047bb: 43 c6 04 2c 00              	movb	$0, (%r12,%r13)
1000047c0: f6 45 c0 01                 	testb	$1, -64(%rbp)
1000047c4: 74 06                       	je	6 <_main+0x18ec>
1000047c6: 48 8b 5d d0                 	movq	-48(%rbp), %rbx
1000047ca: eb 03                       	jmp	3 <_main+0x18ef>
1000047cc: 48 ff c3                    	incq	%rbx
1000047cf: 4c 8b 65 b0                 	movq	-80(%rbp), %r12
1000047d3: 49 8b 06                    	movq	(%r14), %rax
1000047d6: 4c 89 f7                    	movq	%r14, %rdi
1000047d9: 48 89 de                    	movq	%rbx, %rsi
1000047dc: 4c 89 ea                    	movq	%r13, %rdx
1000047df: ff 50 60                    	callq	*96(%rax)
1000047e2: 48 89 c3                    	movq	%rax, %rbx
1000047e5: f6 45 c0 01                 	testb	$1, -64(%rbp)
1000047e9: 74 09                       	je	9 <_main+0x1914>
1000047eb: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
1000047ef: e8 c2 26 00 00              	callq	9922 <dyld_stub_binder+0x100006eb6>
1000047f4: 4c 39 eb                    	cmpq	%r13, %rbx
1000047f7: 75 28                       	jne	40 <_main+0x1941>
1000047f9: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
1000047fd: 49 29 f7                    	subq	%rsi, %r15
100004800: 4d 85 ff                    	testq	%r15, %r15
100004803: 7e 11                       	jle	17 <_main+0x1936>
100004805: 49 8b 06                    	movq	(%r14), %rax
100004808: 4c 89 f7                    	movq	%r14, %rdi
10000480b: 4c 89 fa                    	movq	%r15, %rdx
10000480e: ff 50 60                    	callq	*96(%rax)
100004811: 4c 39 f8                    	cmpq	%r15, %rax
100004814: 75 0b                       	jne	11 <_main+0x1941>
100004816: 49 c7 44 24 18 00 00 00 00  	movq	$0, 24(%r12)
10000481f: eb 03                       	jmp	3 <_main+0x1944>
100004821: 45 31 f6                    	xorl	%r14d, %r14d
100004824: 4c 89 f0                    	movq	%r14, %rax
100004827: 48 83 c4 38                 	addq	$56, %rsp
10000482b: 5b                          	popq	%rbx
10000482c: 41 5c                       	popq	%r12
10000482e: 41 5d                       	popq	%r13
100004830: 41 5e                       	popq	%r14
100004832: 41 5f                       	popq	%r15
100004834: 5d                          	popq	%rbp
100004835: c3                          	retq
100004836: 48 89 c3                    	movq	%rax, %rbx
100004839: f6 45 c0 01                 	testb	$1, -64(%rbp)
10000483d: 74 09                       	je	9 <_main+0x1968>
10000483f: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
100004843: e8 6e 26 00 00              	callq	9838 <dyld_stub_binder+0x100006eb6>
100004848: 48 89 df                    	movq	%rbx, %rdi
10000484b: e8 b2 25 00 00              	callq	9650 <dyld_stub_binder+0x100006e02>
100004850: 0f 0b                       	ud2
100004852: 90                          	nop
100004853: 90                          	nop
100004854: 90                          	nop
100004855: 90                          	nop
100004856: 90                          	nop
100004857: 90                          	nop
100004858: 90                          	nop
100004859: 90                          	nop
10000485a: 90                          	nop
10000485b: 90                          	nop
10000485c: 90                          	nop
10000485d: 90                          	nop
10000485e: 90                          	nop
10000485f: 90                          	nop
100004860: 55                          	pushq	%rbp
100004861: 48 89 e5                    	movq	%rsp, %rbp
100004864: 48 8b 05 95 47 00 00        	movq	18325(%rip), %rax
10000486b: 80 38 00                    	cmpb	$0, (%rax)
10000486e: 74 02                       	je	2 <_main+0x1992>
100004870: 5d                          	popq	%rbp
100004871: c3                          	retq
100004872: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004879: 5d                          	popq	%rbp
10000487a: c3                          	retq
10000487b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004880: 55                          	pushq	%rbp
100004881: 48 89 e5                    	movq	%rsp, %rbp
100004884: 48 8b 05 95 47 00 00        	movq	18325(%rip), %rax
10000488b: 80 38 00                    	cmpb	$0, (%rax)
10000488e: 74 02                       	je	2 <_main+0x19b2>
100004890: 5d                          	popq	%rbp
100004891: c3                          	retq
100004892: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004899: 5d                          	popq	%rbp
10000489a: c3                          	retq
10000489b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048a0: 55                          	pushq	%rbp
1000048a1: 48 89 e5                    	movq	%rsp, %rbp
1000048a4: 48 8b 05 8d 47 00 00        	movq	18317(%rip), %rax
1000048ab: 80 38 00                    	cmpb	$0, (%rax)
1000048ae: 74 02                       	je	2 <_main+0x19d2>
1000048b0: 5d                          	popq	%rbp
1000048b1: c3                          	retq
1000048b2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048b9: 5d                          	popq	%rbp
1000048ba: c3                          	retq
1000048bb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048c0: 55                          	pushq	%rbp
1000048c1: 48 89 e5                    	movq	%rsp, %rbp
1000048c4: 48 8b 05 65 47 00 00        	movq	18277(%rip), %rax
1000048cb: 80 38 00                    	cmpb	$0, (%rax)
1000048ce: 74 02                       	je	2 <_main+0x19f2>
1000048d0: 5d                          	popq	%rbp
1000048d1: c3                          	retq
1000048d2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048d9: 5d                          	popq	%rbp
1000048da: c3                          	retq
1000048db: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000048e0: 55                          	pushq	%rbp
1000048e1: 48 89 e5                    	movq	%rsp, %rbp
1000048e4: 48 8b 05 3d 47 00 00        	movq	18237(%rip), %rax
1000048eb: 80 38 00                    	cmpb	$0, (%rax)
1000048ee: 74 02                       	je	2 <_main+0x1a12>
1000048f0: 5d                          	popq	%rbp
1000048f1: c3                          	retq
1000048f2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
1000048f9: 5d                          	popq	%rbp
1000048fa: c3                          	retq
1000048fb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004900: 55                          	pushq	%rbp
100004901: 48 89 e5                    	movq	%rsp, %rbp
100004904: 48 8b 05 fd 46 00 00        	movq	18173(%rip), %rax
10000490b: 80 38 00                    	cmpb	$0, (%rax)
10000490e: 74 02                       	je	2 <_main+0x1a32>
100004910: 5d                          	popq	%rbp
100004911: c3                          	retq
100004912: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004919: 5d                          	popq	%rbp
10000491a: c3                          	retq
10000491b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004920: 55                          	pushq	%rbp
100004921: 48 89 e5                    	movq	%rsp, %rbp
100004924: 48 8b 05 e5 46 00 00        	movq	18149(%rip), %rax
10000492b: 80 38 00                    	cmpb	$0, (%rax)
10000492e: 74 02                       	je	2 <_main+0x1a52>
100004930: 5d                          	popq	%rbp
100004931: c3                          	retq
100004932: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004939: 5d                          	popq	%rbp
10000493a: c3                          	retq
10000493b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004940: 55                          	pushq	%rbp
100004941: 48 89 e5                    	movq	%rsp, %rbp
100004944: 48 8b 05 f5 46 00 00        	movq	18165(%rip), %rax
10000494b: 80 38 00                    	cmpb	$0, (%rax)
10000494e: 74 02                       	je	2 <_main+0x1a72>
100004950: 5d                          	popq	%rbp
100004951: c3                          	retq
100004952: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004959: 5d                          	popq	%rbp
10000495a: c3                          	retq
10000495b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004960: 55                          	pushq	%rbp
100004961: 48 89 e5                    	movq	%rsp, %rbp
100004964: 48 8b 05 ad 46 00 00        	movq	18093(%rip), %rax
10000496b: 80 38 00                    	cmpb	$0, (%rax)
10000496e: 74 02                       	je	2 <_main+0x1a92>
100004970: 5d                          	popq	%rbp
100004971: c3                          	retq
100004972: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100004979: 5d                          	popq	%rbp
10000497a: c3                          	retq
10000497b: 90                          	nop
10000497c: 90                          	nop
10000497d: 90                          	nop
10000497e: 90                          	nop
10000497f: 90                          	nop

0000000100004980 __ZN14ModelInterfaceC2Ev:
100004980: 55                          	pushq	%rbp
100004981: 48 89 e5                    	movq	%rsp, %rbp
100004984: 48 8d 05 3d 47 00 00        	leaq	18237(%rip), %rax
10000498b: 48 89 07                    	movq	%rax, (%rdi)
10000498e: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100004992: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
100004997: 5d                          	popq	%rbp
100004998: c3                          	retq
100004999: 0f 1f 80 00 00 00 00        	nopl	(%rax)

00000001000049a0 __ZN14ModelInterfaceC1Ev:
1000049a0: 55                          	pushq	%rbp
1000049a1: 48 89 e5                    	movq	%rsp, %rbp
1000049a4: 48 8d 05 1d 47 00 00        	leaq	18205(%rip), %rax
1000049ab: 48 89 07                    	movq	%rax, (%rdi)
1000049ae: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000049b2: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
1000049b7: 5d                          	popq	%rbp
1000049b8: c3                          	retq
1000049b9: 0f 1f 80 00 00 00 00        	nopl	(%rax)

00000001000049c0 __ZN14ModelInterfaceD2Ev:
1000049c0: 55                          	pushq	%rbp
1000049c1: 48 89 e5                    	movq	%rsp, %rbp
1000049c4: 53                          	pushq	%rbx
1000049c5: 50                          	pushq	%rax
1000049c6: 48 89 fb                    	movq	%rdi, %rbx
1000049c9: 48 8d 05 f8 46 00 00        	leaq	18168(%rip), %rax
1000049d0: 48 89 07                    	movq	%rax, (%rdi)
1000049d3: 48 8b 7f 28                 	movq	40(%rdi), %rdi
1000049d7: 48 85 ff                    	testq	%rdi, %rdi
1000049da: 74 05                       	je	5 <__ZN14ModelInterfaceD2Ev+0x21>
1000049dc: e8 d5 24 00 00              	callq	9429 <dyld_stub_binder+0x100006eb6>
1000049e1: 48 8b 7b 30                 	movq	48(%rbx), %rdi
1000049e5: 48 83 c4 08                 	addq	$8, %rsp
1000049e9: 48 85 ff                    	testq	%rdi, %rdi
1000049ec: 74 07                       	je	7 <__ZN14ModelInterfaceD2Ev+0x35>
1000049ee: 5b                          	popq	%rbx
1000049ef: 5d                          	popq	%rbp
1000049f0: e9 c1 24 00 00              	jmp	9409 <dyld_stub_binder+0x100006eb6>
1000049f5: 5b                          	popq	%rbx
1000049f6: 5d                          	popq	%rbp
1000049f7: c3                          	retq
1000049f8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100004a00 __ZN14ModelInterfaceD1Ev:
100004a00: 55                          	pushq	%rbp
100004a01: 48 89 e5                    	movq	%rsp, %rbp
100004a04: 53                          	pushq	%rbx
100004a05: 50                          	pushq	%rax
100004a06: 48 89 fb                    	movq	%rdi, %rbx
100004a09: 48 8d 05 b8 46 00 00        	leaq	18104(%rip), %rax
100004a10: 48 89 07                    	movq	%rax, (%rdi)
100004a13: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100004a17: 48 85 ff                    	testq	%rdi, %rdi
100004a1a: 74 05                       	je	5 <__ZN14ModelInterfaceD1Ev+0x21>
100004a1c: e8 95 24 00 00              	callq	9365 <dyld_stub_binder+0x100006eb6>
100004a21: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100004a25: 48 83 c4 08                 	addq	$8, %rsp
100004a29: 48 85 ff                    	testq	%rdi, %rdi
100004a2c: 74 07                       	je	7 <__ZN14ModelInterfaceD1Ev+0x35>
100004a2e: 5b                          	popq	%rbx
100004a2f: 5d                          	popq	%rbp
100004a30: e9 81 24 00 00              	jmp	9345 <dyld_stub_binder+0x100006eb6>
100004a35: 5b                          	popq	%rbx
100004a36: 5d                          	popq	%rbp
100004a37: c3                          	retq
100004a38: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100004a40 __ZN14ModelInterfaceD0Ev:
100004a40: 55                          	pushq	%rbp
100004a41: 48 89 e5                    	movq	%rsp, %rbp
100004a44: 53                          	pushq	%rbx
100004a45: 50                          	pushq	%rax
100004a46: 48 89 fb                    	movq	%rdi, %rbx
100004a49: 48 8d 05 78 46 00 00        	leaq	18040(%rip), %rax
100004a50: 48 89 07                    	movq	%rax, (%rdi)
100004a53: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100004a57: 48 85 ff                    	testq	%rdi, %rdi
100004a5a: 74 05                       	je	5 <__ZN14ModelInterfaceD0Ev+0x21>
100004a5c: e8 55 24 00 00              	callq	9301 <dyld_stub_binder+0x100006eb6>
100004a61: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100004a65: 48 85 ff                    	testq	%rdi, %rdi
100004a68: 74 05                       	je	5 <__ZN14ModelInterfaceD0Ev+0x2f>
100004a6a: e8 47 24 00 00              	callq	9287 <dyld_stub_binder+0x100006eb6>
100004a6f: 48 89 df                    	movq	%rbx, %rdi
100004a72: 48 83 c4 08                 	addq	$8, %rsp
100004a76: 5b                          	popq	%rbx
100004a77: 5d                          	popq	%rbp
100004a78: e9 39 24 00 00              	jmp	9273 <dyld_stub_binder+0x100006eb6>
100004a7d: 0f 1f 00                    	nopl	(%rax)

0000000100004a80 __ZN14ModelInterface7forwardEv:
100004a80: 55                          	pushq	%rbp
100004a81: 48 89 e5                    	movq	%rsp, %rbp
100004a84: 5d                          	popq	%rbp
100004a85: c3                          	retq
100004a86: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)

0000000100004a90 __ZN14ModelInterface12input_bufferEv:
100004a90: 55                          	pushq	%rbp
100004a91: 48 89 e5                    	movq	%rsp, %rbp
100004a94: 0f b6 47 24                 	movzbl	36(%rdi), %eax
100004a98: 48 8b 44 c7 28              	movq	40(%rdi,%rax,8), %rax
100004a9d: 5d                          	popq	%rbp
100004a9e: c3                          	retq
100004a9f: 90                          	nop

0000000100004aa0 __ZN14ModelInterface13output_bufferEv:
100004aa0: 55                          	pushq	%rbp
100004aa1: 48 89 e5                    	movq	%rsp, %rbp
100004aa4: 31 c0                       	xorl	%eax, %eax
100004aa6: 80 7f 24 00                 	cmpb	$0, 36(%rdi)
100004aaa: 0f 94 c0                    	sete	%al
100004aad: 48 8b 44 c7 28              	movq	40(%rdi,%rax,8), %rax
100004ab2: 5d                          	popq	%rbp
100004ab3: c3                          	retq
100004ab4: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004abe: 66 90                       	nop

0000000100004ac0 __ZN14ModelInterface11init_bufferEj:
100004ac0: 55                          	pushq	%rbp
100004ac1: 48 89 e5                    	movq	%rsp, %rbp
100004ac4: 41 57                       	pushq	%r15
100004ac6: 41 56                       	pushq	%r14
100004ac8: 41 54                       	pushq	%r12
100004aca: 53                          	pushq	%rbx
100004acb: 41 89 f7                    	movl	%esi, %r15d
100004ace: 48 89 fb                    	movq	%rdi, %rbx
100004ad1: c6 47 24 00                 	movb	$0, 36(%rdi)
100004ad5: 41 89 f6                    	movl	%esi, %r14d
100004ad8: 4c 89 f7                    	movq	%r14, %rdi
100004adb: e8 dc 23 00 00              	callq	9180 <dyld_stub_binder+0x100006ebc>
100004ae0: 49 89 c4                    	movq	%rax, %r12
100004ae3: 48 89 43 28                 	movq	%rax, 40(%rbx)
100004ae7: 4c 89 f7                    	movq	%r14, %rdi
100004aea: e8 cd 23 00 00              	callq	9165 <dyld_stub_binder+0x100006ebc>
100004aef: 48 89 43 30                 	movq	%rax, 48(%rbx)
100004af3: 45 85 ff                    	testl	%r15d, %r15d
100004af6: 0f 84 44 01 00 00           	je	324 <__ZN14ModelInterface11init_bufferEj+0x180>
100004afc: 41 c6 04 24 00              	movb	$0, (%r12)
100004b01: 41 83 ff 01                 	cmpl	$1, %r15d
100004b05: 0f 84 95 00 00 00           	je	149 <__ZN14ModelInterface11init_bufferEj+0xe0>
100004b0b: 41 8d 46 ff                 	leal	-1(%r14), %eax
100004b0f: 49 8d 56 fe                 	leaq	-2(%r14), %rdx
100004b13: 83 e0 07                    	andl	$7, %eax
100004b16: b9 01 00 00 00              	movl	$1, %ecx
100004b1b: 48 83 fa 07                 	cmpq	$7, %rdx
100004b1f: 72 63                       	jb	99 <__ZN14ModelInterface11init_bufferEj+0xc4>
100004b21: 48 89 c2                    	movq	%rax, %rdx
100004b24: 48 f7 d2                    	notq	%rdx
100004b27: 4c 01 f2                    	addq	%r14, %rdx
100004b2a: 31 c9                       	xorl	%ecx, %ecx
100004b2c: 0f 1f 40 00                 	nopl	(%rax)
100004b30: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b34: c6 44 0e 01 00              	movb	$0, 1(%rsi,%rcx)
100004b39: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b3d: c6 44 0e 02 00              	movb	$0, 2(%rsi,%rcx)
100004b42: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b46: c6 44 0e 03 00              	movb	$0, 3(%rsi,%rcx)
100004b4b: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b4f: c6 44 0e 04 00              	movb	$0, 4(%rsi,%rcx)
100004b54: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b58: c6 44 0e 05 00              	movb	$0, 5(%rsi,%rcx)
100004b5d: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b61: c6 44 0e 06 00              	movb	$0, 6(%rsi,%rcx)
100004b66: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b6a: c6 44 0e 07 00              	movb	$0, 7(%rsi,%rcx)
100004b6f: 48 8b 73 28                 	movq	40(%rbx), %rsi
100004b73: c6 44 0e 08 00              	movb	$0, 8(%rsi,%rcx)
100004b78: 48 83 c1 08                 	addq	$8, %rcx
100004b7c: 48 39 ca                    	cmpq	%rcx, %rdx
100004b7f: 75 af                       	jne	-81 <__ZN14ModelInterface11init_bufferEj+0x70>
100004b81: 48 ff c1                    	incq	%rcx
100004b84: 48 85 c0                    	testq	%rax, %rax
100004b87: 74 17                       	je	23 <__ZN14ModelInterface11init_bufferEj+0xe0>
100004b89: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004b90: 48 8b 53 28                 	movq	40(%rbx), %rdx
100004b94: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004b98: 48 ff c1                    	incq	%rcx
100004b9b: 48 ff c8                    	decq	%rax
100004b9e: 75 f0                       	jne	-16 <__ZN14ModelInterface11init_bufferEj+0xd0>
100004ba0: 45 85 ff                    	testl	%r15d, %r15d
100004ba3: 0f 84 97 00 00 00           	je	151 <__ZN14ModelInterface11init_bufferEj+0x180>
100004ba9: 49 8d 4e ff                 	leaq	-1(%r14), %rcx
100004bad: 44 89 f0                    	movl	%r14d, %eax
100004bb0: 83 e0 07                    	andl	$7, %eax
100004bb3: 48 83 f9 07                 	cmpq	$7, %rcx
100004bb7: 73 0c                       	jae	12 <__ZN14ModelInterface11init_bufferEj+0x105>
100004bb9: 31 c9                       	xorl	%ecx, %ecx
100004bbb: 48 85 c0                    	testq	%rax, %rax
100004bbe: 75 70                       	jne	112 <__ZN14ModelInterface11init_bufferEj+0x170>
100004bc0: e9 7b 00 00 00              	jmp	123 <__ZN14ModelInterface11init_bufferEj+0x180>
100004bc5: 49 29 c6                    	subq	%rax, %r14
100004bc8: 31 c9                       	xorl	%ecx, %ecx
100004bca: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100004bd0: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004bd4: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004bd8: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004bdc: c6 44 0a 01 00              	movb	$0, 1(%rdx,%rcx)
100004be1: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004be5: c6 44 0a 02 00              	movb	$0, 2(%rdx,%rcx)
100004bea: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004bee: c6 44 0a 03 00              	movb	$0, 3(%rdx,%rcx)
100004bf3: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004bf7: c6 44 0a 04 00              	movb	$0, 4(%rdx,%rcx)
100004bfc: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004c00: c6 44 0a 05 00              	movb	$0, 5(%rdx,%rcx)
100004c05: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004c09: c6 44 0a 06 00              	movb	$0, 6(%rdx,%rcx)
100004c0e: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004c12: c6 44 0a 07 00              	movb	$0, 7(%rdx,%rcx)
100004c17: 48 83 c1 08                 	addq	$8, %rcx
100004c1b: 49 39 ce                    	cmpq	%rcx, %r14
100004c1e: 75 b0                       	jne	-80 <__ZN14ModelInterface11init_bufferEj+0x110>
100004c20: 48 85 c0                    	testq	%rax, %rax
100004c23: 74 1b                       	je	27 <__ZN14ModelInterface11init_bufferEj+0x180>
100004c25: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004c2f: 90                          	nop
100004c30: 48 8b 53 30                 	movq	48(%rbx), %rdx
100004c34: c6 04 0a 00                 	movb	$0, (%rdx,%rcx)
100004c38: 48 ff c1                    	incq	%rcx
100004c3b: 48 ff c8                    	decq	%rax
100004c3e: 75 f0                       	jne	-16 <__ZN14ModelInterface11init_bufferEj+0x170>
100004c40: 5b                          	popq	%rbx
100004c41: 41 5c                       	popq	%r12
100004c43: 41 5e                       	popq	%r14
100004c45: 41 5f                       	popq	%r15
100004c47: 5d                          	popq	%rbp
100004c48: c3                          	retq
100004c49: 0f 1f 80 00 00 00 00        	nopl	(%rax)

0000000100004c50 __ZN14ModelInterface11swap_bufferEv:
100004c50: 55                          	pushq	%rbp
100004c51: 48 89 e5                    	movq	%rsp, %rbp
100004c54: 80 77 24 01                 	xorb	$1, 36(%rdi)
100004c58: 5d                          	popq	%rbp
100004c59: c3                          	retq
100004c5a: 90                          	nop
100004c5b: 90                          	nop
100004c5c: 90                          	nop
100004c5d: 90                          	nop
100004c5e: 90                          	nop
100004c5f: 90                          	nop

0000000100004c60 __Z4ReLUPaS_j:
100004c60: 55                          	pushq	%rbp
100004c61: 48 89 e5                    	movq	%rsp, %rbp
100004c64: 83 fa 04                    	cmpl	$4, %edx
100004c67: 0f 82 88 00 00 00           	jb	136 <__Z4ReLUPaS_j+0x95>
100004c6d: 8d 42 fc                    	leal	-4(%rdx), %eax
100004c70: 41 89 c2                    	movl	%eax, %r10d
100004c73: 41 c1 ea 02                 	shrl	$2, %r10d
100004c77: 41 ff c2                    	incl	%r10d
100004c7a: 41 83 fa 1f                 	cmpl	$31, %r10d
100004c7e: 76 24                       	jbe	36 <__Z4ReLUPaS_j+0x44>
100004c80: 83 e0 fc                    	andl	$-4, %eax
100004c83: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100004c87: 48 83 c1 04                 	addq	$4, %rcx
100004c8b: 48 39 f9                    	cmpq	%rdi, %rcx
100004c8e: 0f 86 78 02 00 00           	jbe	632 <__Z4ReLUPaS_j+0x2ac>
100004c94: 48 01 f8                    	addq	%rdi, %rax
100004c97: 48 83 c0 04                 	addq	$4, %rax
100004c9b: 48 39 f0                    	cmpq	%rsi, %rax
100004c9e: 0f 86 68 02 00 00           	jbe	616 <__Z4ReLUPaS_j+0x2ac>
100004ca4: 89 d0                       	movl	%edx, %eax
100004ca6: 45 31 c0                    	xorl	%r8d, %r8d
100004ca9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004cb0: 0f b6 0e                    	movzbl	(%rsi), %ecx
100004cb3: 84 c9                       	testb	%cl, %cl
100004cb5: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004cb9: 88 0f                       	movb	%cl, (%rdi)
100004cbb: 0f b6 4e 01                 	movzbl	1(%rsi), %ecx
100004cbf: 84 c9                       	testb	%cl, %cl
100004cc1: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004cc5: 88 4f 01                    	movb	%cl, 1(%rdi)
100004cc8: 0f b6 4e 02                 	movzbl	2(%rsi), %ecx
100004ccc: 84 c9                       	testb	%cl, %cl
100004cce: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004cd2: 88 4f 02                    	movb	%cl, 2(%rdi)
100004cd5: 0f b6 4e 03                 	movzbl	3(%rsi), %ecx
100004cd9: 84 c9                       	testb	%cl, %cl
100004cdb: 41 0f 48 c8                 	cmovsl	%r8d, %ecx
100004cdf: 88 4f 03                    	movb	%cl, 3(%rdi)
100004ce2: 48 83 c7 04                 	addq	$4, %rdi
100004ce6: 48 83 c6 04                 	addq	$4, %rsi
100004cea: 83 c0 fc                    	addl	$-4, %eax
100004ced: 83 f8 03                    	cmpl	$3, %eax
100004cf0: 77 be                       	ja	-66 <__Z4ReLUPaS_j+0x50>
100004cf2: 83 e2 03                    	andl	$3, %edx
100004cf5: 85 d2                       	testl	%edx, %edx
100004cf7: 0f 84 0a 02 00 00           	je	522 <__Z4ReLUPaS_j+0x2a7>
100004cfd: 8d 42 ff                    	leal	-1(%rdx), %eax
100004d00: 4c 8d 50 01                 	leaq	1(%rax), %r10
100004d04: 49 83 fa 7f                 	cmpq	$127, %r10
100004d08: 0f 86 2e 01 00 00           	jbe	302 <__Z4ReLUPaS_j+0x1dc>
100004d0e: 48 8d 0c 06                 	leaq	(%rsi,%rax), %rcx
100004d12: 48 83 c1 01                 	addq	$1, %rcx
100004d16: 48 39 cf                    	cmpq	%rcx, %rdi
100004d19: 73 10                       	jae	16 <__Z4ReLUPaS_j+0xcb>
100004d1b: 48 01 f8                    	addq	%rdi, %rax
100004d1e: 48 83 c0 01                 	addq	$1, %rax
100004d22: 48 39 c6                    	cmpq	%rax, %rsi
100004d25: 0f 82 11 01 00 00           	jb	273 <__Z4ReLUPaS_j+0x1dc>
100004d2b: 4d 89 d0                    	movq	%r10, %r8
100004d2e: 49 83 e0 80                 	andq	$-128, %r8
100004d32: 49 8d 40 80                 	leaq	-128(%r8), %rax
100004d36: 48 89 c1                    	movq	%rax, %rcx
100004d39: 48 c1 e9 07                 	shrq	$7, %rcx
100004d3d: 48 ff c1                    	incq	%rcx
100004d40: 41 89 c9                    	movl	%ecx, %r9d
100004d43: 41 83 e1 01                 	andl	$1, %r9d
100004d47: 48 85 c0                    	testq	%rax, %rax
100004d4a: 0f 84 0f 09 00 00           	je	2319 <__Z4ReLUPaS_j+0x9ff>
100004d50: 4c 89 c8                    	movq	%r9, %rax
100004d53: 48 29 c8                    	subq	%rcx, %rax
100004d56: 31 c9                       	xorl	%ecx, %ecx
100004d58: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100004d5c: 0f 1f 40 00                 	nopl	(%rax)
100004d60: c4 e2 7d 3c 0c 0e           	vpmaxsb	(%rsi,%rcx), %ymm0, %ymm1
100004d66: c4 e2 7d 3c 54 0e 20        	vpmaxsb	32(%rsi,%rcx), %ymm0, %ymm2
100004d6d: c4 e2 7d 3c 5c 0e 40        	vpmaxsb	64(%rsi,%rcx), %ymm0, %ymm3
100004d74: c4 e2 7d 3c 64 0e 60        	vpmaxsb	96(%rsi,%rcx), %ymm0, %ymm4
100004d7b: c5 fe 7f 0c 0f              	vmovdqu	%ymm1, (%rdi,%rcx)
100004d80: c5 fe 7f 54 0f 20           	vmovdqu	%ymm2, 32(%rdi,%rcx)
100004d86: c5 fe 7f 5c 0f 40           	vmovdqu	%ymm3, 64(%rdi,%rcx)
100004d8c: c5 fe 7f 64 0f 60           	vmovdqu	%ymm4, 96(%rdi,%rcx)
100004d92: c4 e2 7d 3c 8c 0e 80 00 00 00       	vpmaxsb	128(%rsi,%rcx), %ymm0, %ymm1
100004d9c: c4 e2 7d 3c 94 0e a0 00 00 00       	vpmaxsb	160(%rsi,%rcx), %ymm0, %ymm2
100004da6: c4 e2 7d 3c 9c 0e c0 00 00 00       	vpmaxsb	192(%rsi,%rcx), %ymm0, %ymm3
100004db0: c4 e2 7d 3c a4 0e e0 00 00 00       	vpmaxsb	224(%rsi,%rcx), %ymm0, %ymm4
100004dba: c5 fe 7f 8c 0f 80 00 00 00  	vmovdqu	%ymm1, 128(%rdi,%rcx)
100004dc3: c5 fe 7f 94 0f a0 00 00 00  	vmovdqu	%ymm2, 160(%rdi,%rcx)
100004dcc: c5 fe 7f 9c 0f c0 00 00 00  	vmovdqu	%ymm3, 192(%rdi,%rcx)
100004dd5: c5 fe 7f a4 0f e0 00 00 00  	vmovdqu	%ymm4, 224(%rdi,%rcx)
100004dde: 48 81 c1 00 01 00 00        	addq	$256, %rcx
100004de5: 48 83 c0 02                 	addq	$2, %rax
100004de9: 0f 85 71 ff ff ff           	jne	-143 <__Z4ReLUPaS_j+0x100>
100004def: 4d 85 c9                    	testq	%r9, %r9
100004df2: 74 36                       	je	54 <__Z4ReLUPaS_j+0x1ca>
100004df4: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100004df8: c4 e2 7d 3c 0c 0e           	vpmaxsb	(%rsi,%rcx), %ymm0, %ymm1
100004dfe: c4 e2 7d 3c 54 0e 20        	vpmaxsb	32(%rsi,%rcx), %ymm0, %ymm2
100004e05: c4 e2 7d 3c 5c 0e 40        	vpmaxsb	64(%rsi,%rcx), %ymm0, %ymm3
100004e0c: c4 e2 7d 3c 44 0e 60        	vpmaxsb	96(%rsi,%rcx), %ymm0, %ymm0
100004e13: c5 fe 7f 0c 0f              	vmovdqu	%ymm1, (%rdi,%rcx)
100004e18: c5 fe 7f 54 0f 20           	vmovdqu	%ymm2, 32(%rdi,%rcx)
100004e1e: c5 fe 7f 5c 0f 40           	vmovdqu	%ymm3, 64(%rdi,%rcx)
100004e24: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
100004e2a: 4d 39 c2                    	cmpq	%r8, %r10
100004e2d: 0f 84 d4 00 00 00           	je	212 <__Z4ReLUPaS_j+0x2a7>
100004e33: 44 29 c2                    	subl	%r8d, %edx
100004e36: 4c 01 c6                    	addq	%r8, %rsi
100004e39: 4c 01 c7                    	addq	%r8, %rdi
100004e3c: 44 8d 42 ff                 	leal	-1(%rdx), %r8d
100004e40: f6 c2 07                    	testb	$7, %dl
100004e43: 74 38                       	je	56 <__Z4ReLUPaS_j+0x21d>
100004e45: 41 89 d2                    	movl	%edx, %r10d
100004e48: 41 83 e2 07                 	andl	$7, %r10d
100004e4c: 45 31 c9                    	xorl	%r9d, %r9d
100004e4f: 31 c9                       	xorl	%ecx, %ecx
100004e51: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004e5b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100004e60: 0f b6 04 0e                 	movzbl	(%rsi,%rcx), %eax
100004e64: 84 c0                       	testb	%al, %al
100004e66: 41 0f 48 c1                 	cmovsl	%r9d, %eax
100004e6a: 88 04 0f                    	movb	%al, (%rdi,%rcx)
100004e6d: 48 ff c1                    	incq	%rcx
100004e70: 41 39 ca                    	cmpl	%ecx, %r10d
100004e73: 75 eb                       	jne	-21 <__Z4ReLUPaS_j+0x200>
100004e75: 29 ca                       	subl	%ecx, %edx
100004e77: 48 01 ce                    	addq	%rcx, %rsi
100004e7a: 48 01 cf                    	addq	%rcx, %rdi
100004e7d: 41 83 f8 07                 	cmpl	$7, %r8d
100004e81: 0f 82 80 00 00 00           	jb	128 <__Z4ReLUPaS_j+0x2a7>
100004e87: 41 89 d0                    	movl	%edx, %r8d
100004e8a: 31 c9                       	xorl	%ecx, %ecx
100004e8c: 31 d2                       	xorl	%edx, %edx
100004e8e: 66 90                       	nop
100004e90: 0f b6 04 16                 	movzbl	(%rsi,%rdx), %eax
100004e94: 84 c0                       	testb	%al, %al
100004e96: 0f 48 c1                    	cmovsl	%ecx, %eax
100004e99: 88 04 17                    	movb	%al, (%rdi,%rdx)
100004e9c: 0f b6 44 16 01              	movzbl	1(%rsi,%rdx), %eax
100004ea1: 84 c0                       	testb	%al, %al
100004ea3: 0f 48 c1                    	cmovsl	%ecx, %eax
100004ea6: 88 44 17 01                 	movb	%al, 1(%rdi,%rdx)
100004eaa: 0f b6 44 16 02              	movzbl	2(%rsi,%rdx), %eax
100004eaf: 84 c0                       	testb	%al, %al
100004eb1: 0f 48 c1                    	cmovsl	%ecx, %eax
100004eb4: 88 44 17 02                 	movb	%al, 2(%rdi,%rdx)
100004eb8: 0f b6 44 16 03              	movzbl	3(%rsi,%rdx), %eax
100004ebd: 84 c0                       	testb	%al, %al
100004ebf: 0f 48 c1                    	cmovsl	%ecx, %eax
100004ec2: 88 44 17 03                 	movb	%al, 3(%rdi,%rdx)
100004ec6: 0f b6 44 16 04              	movzbl	4(%rsi,%rdx), %eax
100004ecb: 84 c0                       	testb	%al, %al
100004ecd: 0f 48 c1                    	cmovsl	%ecx, %eax
100004ed0: 88 44 17 04                 	movb	%al, 4(%rdi,%rdx)
100004ed4: 0f b6 44 16 05              	movzbl	5(%rsi,%rdx), %eax
100004ed9: 84 c0                       	testb	%al, %al
100004edb: 0f 48 c1                    	cmovsl	%ecx, %eax
100004ede: 88 44 17 05                 	movb	%al, 5(%rdi,%rdx)
100004ee2: 0f b6 44 16 06              	movzbl	6(%rsi,%rdx), %eax
100004ee7: 84 c0                       	testb	%al, %al
100004ee9: 0f 48 c1                    	cmovsl	%ecx, %eax
100004eec: 88 44 17 06                 	movb	%al, 6(%rdi,%rdx)
100004ef0: 0f b6 44 16 07              	movzbl	7(%rsi,%rdx), %eax
100004ef5: 84 c0                       	testb	%al, %al
100004ef7: 0f 48 c1                    	cmovsl	%ecx, %eax
100004efa: 88 44 17 07                 	movb	%al, 7(%rdi,%rdx)
100004efe: 48 83 c2 08                 	addq	$8, %rdx
100004f02: 41 39 d0                    	cmpl	%edx, %r8d
100004f05: 75 89                       	jne	-119 <__Z4ReLUPaS_j+0x230>
100004f07: 5d                          	popq	%rbp
100004f08: c5 f8 77                    	vzeroupper
100004f0b: c3                          	retq
100004f0c: 45 89 d0                    	movl	%r10d, %r8d
100004f0f: 41 83 e0 e0                 	andl	$-32, %r8d
100004f13: 49 8d 40 e0                 	leaq	-32(%r8), %rax
100004f17: 48 89 c1                    	movq	%rax, %rcx
100004f1a: 48 c1 e9 05                 	shrq	$5, %rcx
100004f1e: 48 ff c1                    	incq	%rcx
100004f21: 41 89 c9                    	movl	%ecx, %r9d
100004f24: 41 83 e1 01                 	andl	$1, %r9d
100004f28: 48 85 c0                    	testq	%rax, %rax
100004f2b: 0f 84 3e 07 00 00           	je	1854 <__Z4ReLUPaS_j+0xa0f>
100004f31: 4c 89 c8                    	movq	%r9, %rax
100004f34: 48 29 c8                    	subq	%rcx, %rax
100004f37: 31 c9                       	xorl	%ecx, %ecx
100004f39: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100004f40: c5 7a 6f 34 0e              	vmovdqu	(%rsi,%rcx), %xmm14
100004f45: c5 7a 6f 7c 0e 10           	vmovdqu	16(%rsi,%rcx), %xmm15
100004f4b: c5 fa 6f 54 0e 20           	vmovdqu	32(%rsi,%rcx), %xmm2
100004f51: c5 fa 6f 5c 0e 30           	vmovdqu	48(%rsi,%rcx), %xmm3
100004f57: c5 79 6f 1d e1 21 00 00     	vmovdqa	8673(%rip), %xmm11
100004f5f: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100004f64: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
100004f69: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100004f6d: c5 79 6f 05 db 21 00 00     	vmovdqa	8667(%rip), %xmm8
100004f75: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
100004f7a: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
100004f7f: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004f83: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
100004f89: c5 fa 6f 64 0e 70           	vmovdqu	112(%rsi,%rcx), %xmm4
100004f8f: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100004f94: c4 e3 fd 00 6c 0e 60 4e     	vpermq	$78, 96(%rsi,%rcx), %ymm5
100004f9c: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004fa2: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004fa7: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
100004fab: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100004fb1: c5 fa 6f 74 0e 50           	vmovdqu	80(%rsi,%rcx), %xmm6
100004fb7: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
100004fbc: c4 63 fd 00 4c 0e 40 4e     	vpermq	$78, 64(%rsi,%rcx), %ymm9
100004fc4: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
100004fca: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
100004fcf: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100004fd4: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004fda: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100004fe0: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004fe6: c5 79 6f 05 72 21 00 00     	vmovdqa	8562(%rip), %xmm8
100004fee: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100004ff3: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100004ff8: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
100004ffc: c5 79 6f 1d 6c 21 00 00     	vmovdqa	8556(%rip), %xmm11
100005004: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
100005009: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000500e: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
100005012: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
100005018: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
10000501d: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
100005022: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005026: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
10000502c: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
100005031: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
100005036: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
10000503a: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005040: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005046: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
10000504c: c5 79 6f 1d 2c 21 00 00     	vmovdqa	8492(%rip), %xmm11
100005054: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005059: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
10000505e: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
100005062: c5 f9 6f 0d 26 21 00 00     	vmovdqa	8486(%rip), %xmm1
10000506a: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
10000506e: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100005073: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100005078: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
10000507c: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100005082: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005087: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
10000508c: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005090: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005096: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
10000509b: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
1000050a0: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
1000050a4: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000050aa: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000050b0: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
1000050b6: c5 f9 6f 0d e2 20 00 00     	vmovdqa	8418(%rip), %xmm1
1000050be: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
1000050c3: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
1000050c8: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
1000050cc: c5 f9 6f 15 dc 20 00 00     	vmovdqa	8412(%rip), %xmm2
1000050d4: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
1000050d8: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
1000050dd: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
1000050e2: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000050e6: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
1000050ec: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
1000050f1: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
1000050f6: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000050fa: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005100: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
100005105: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
10000510a: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
10000510e: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100005114: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
10000511a: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
100005120: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
100005125: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
10000512a: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
10000512f: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
100005134: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100005139: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
10000513d: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005141: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005145: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100005149: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
10000514d: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100005151: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005155: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100005159: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
10000515f: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
100005165: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
10000516b: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005171: c5 fe 7f 4c 0f 40           	vmovdqu	%ymm1, 64(%rdi,%rcx)
100005177: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
10000517d: c5 fe 7f 6c 0f 20           	vmovdqu	%ymm5, 32(%rdi,%rcx)
100005183: c5 fe 7f 14 0f              	vmovdqu	%ymm2, (%rdi,%rcx)
100005188: c5 7a 6f a4 0e 80 00 00 00  	vmovdqu	128(%rsi,%rcx), %xmm12
100005191: c5 7a 6f ac 0e 90 00 00 00  	vmovdqu	144(%rsi,%rcx), %xmm13
10000519a: c5 7a 6f b4 0e a0 00 00 00  	vmovdqu	160(%rsi,%rcx), %xmm14
1000051a3: c5 fa 6f 9c 0e b0 00 00 00  	vmovdqu	176(%rsi,%rcx), %xmm3
1000051ac: c5 f9 6f 05 8c 1f 00 00     	vmovdqa	8076(%rip), %xmm0
1000051b4: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
1000051b9: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
1000051be: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
1000051c2: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
1000051c6: c5 f9 6f 05 82 1f 00 00     	vmovdqa	8066(%rip), %xmm0
1000051ce: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
1000051d3: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
1000051d8: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
1000051dc: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
1000051e0: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
1000051e6: c5 fa 6f a4 0e f0 00 00 00  	vmovdqu	240(%rsi,%rcx), %xmm4
1000051ef: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
1000051f4: c4 e3 fd 00 ac 0e e0 00 00 00 4e    	vpermq	$78, 224(%rsi,%rcx), %ymm5
1000051ff: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005205: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
10000520a: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000520e: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
100005214: c5 fa 6f b4 0e d0 00 00 00  	vmovdqu	208(%rsi,%rcx), %xmm6
10000521d: c4 e3 fd 00 bc 0e c0 00 00 00 4e    	vpermq	$78, 192(%rsi,%rcx), %ymm7
100005228: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
10000522d: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
100005233: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
100005238: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
10000523c: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005242: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
100005248: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
10000524e: c5 79 6f 3d 0a 1f 00 00     	vmovdqa	7946(%rip), %xmm15
100005256: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
10000525b: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
100005260: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
100005264: c5 f9 6f 05 04 1f 00 00     	vmovdqa	7940(%rip), %xmm0
10000526c: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100005271: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005276: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000527a: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
100005280: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
100005285: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
10000528a: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000528e: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005294: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005299: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
10000529e: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
1000052a2: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000052a8: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000052ae: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
1000052b4: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
1000052b9: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
1000052be: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
1000052c2: c5 f9 6f 05 c6 1e 00 00     	vmovdqa	7878(%rip), %xmm0
1000052ca: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000052cf: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000052d4: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000052d8: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
1000052de: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
1000052e3: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
1000052e8: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000052ec: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
1000052f1: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
1000052f6: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
1000052fa: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005300: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005306: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
10000530c: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
100005312: c5 79 6f 3d 86 1e 00 00     	vmovdqa	7814(%rip), %xmm15
10000531a: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
10000531f: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
100005324: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005328: c5 f9 6f 05 80 1e 00 00     	vmovdqa	7808(%rip), %xmm0
100005330: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
100005335: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
10000533a: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000533e: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
100005344: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
100005349: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
10000534e: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005352: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
100005357: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
10000535c: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005360: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005366: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
10000536c: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100005372: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
100005378: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
10000537d: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
100005382: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
100005387: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
10000538c: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100005390: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005394: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005398: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
10000539c: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000053a0: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000053a4: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000053a8: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000053ac: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
1000053b2: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
1000053b8: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
1000053be: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
1000053c4: c5 fe 7f 8c 0f c0 00 00 00  	vmovdqu	%ymm1, 192(%rdi,%rcx)
1000053cd: c5 fe 7f 84 0f e0 00 00 00  	vmovdqu	%ymm0, 224(%rdi,%rcx)
1000053d6: c5 fe 7f 9c 0f a0 00 00 00  	vmovdqu	%ymm3, 160(%rdi,%rcx)
1000053df: c5 fe 7f 94 0f 80 00 00 00  	vmovdqu	%ymm2, 128(%rdi,%rcx)
1000053e8: 48 81 c1 00 01 00 00        	addq	$256, %rcx
1000053ef: 48 83 c0 02                 	addq	$2, %rax
1000053f3: 0f 85 47 fb ff ff           	jne	-1209 <__Z4ReLUPaS_j+0x2e0>
1000053f9: 4d 85 c9                    	testq	%r9, %r9
1000053fc: 0f 84 3e 02 00 00           	je	574 <__Z4ReLUPaS_j+0x9e0>
100005402: c5 7a 6f 14 0e              	vmovdqu	(%rsi,%rcx), %xmm10
100005407: c5 7a 6f 5c 0e 10           	vmovdqu	16(%rsi,%rcx), %xmm11
10000540d: c5 7a 6f 64 0e 20           	vmovdqu	32(%rsi,%rcx), %xmm12
100005413: c5 7a 6f 6c 0e 30           	vmovdqu	48(%rsi,%rcx), %xmm13
100005419: c5 f9 6f 35 1f 1d 00 00     	vmovdqa	7455(%rip), %xmm6
100005421: c4 e2 11 00 e6              	vpshufb	%xmm6, %xmm13, %xmm4
100005426: c4 e2 19 00 ee              	vpshufb	%xmm6, %xmm12, %xmm5
10000542b: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000542f: c5 f9 6f 05 19 1d 00 00     	vmovdqa	7449(%rip), %xmm0
100005437: c4 e2 21 00 e8              	vpshufb	%xmm0, %xmm11, %xmm5
10000543c: c4 e2 29 00 f8              	vpshufb	%xmm0, %xmm10, %xmm7
100005441: c5 c1 62 ed                 	vpunpckldq	%xmm5, %xmm7, %xmm5
100005445: c4 63 51 02 c4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm8
10000544b: c5 7a 6f 74 0e 70           	vmovdqu	112(%rsi,%rcx), %xmm14
100005451: c4 e2 09 00 fe              	vpshufb	%xmm6, %xmm14, %xmm7
100005456: c4 e3 fd 00 6c 0e 60 4e     	vpermq	$78, 96(%rsi,%rcx), %ymm5
10000545e: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005464: c4 e2 51 00 f6              	vpshufb	%xmm6, %xmm5, %xmm6
100005469: c5 c9 62 f7                 	vpunpckldq	%xmm7, %xmm6, %xmm6
10000546d: c4 63 7d 38 ce 01           	vinserti128	$1, %xmm6, %ymm0, %ymm9
100005473: c5 fa 6f 74 0e 50           	vmovdqu	80(%rsi,%rcx), %xmm6
100005479: c4 e2 49 00 c8              	vpshufb	%xmm0, %xmm6, %xmm1
10000547e: c4 e3 fd 00 7c 0e 40 4e     	vpermq	$78, 64(%rsi,%rcx), %ymm7
100005486: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
10000548c: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005491: c5 f9 62 c1                 	vpunpckldq	%xmm1, %xmm0, %xmm0
100005495: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000549b: c4 c3 7d 02 c1 c0           	vpblendd	$192, %ymm9, %ymm0, %ymm0
1000054a1: c4 63 3d 02 c0 f0           	vpblendd	$240, %ymm0, %ymm8, %ymm8
1000054a7: c5 f9 6f 05 b1 1c 00 00     	vmovdqa	7345(%rip), %xmm0
1000054af: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000054b4: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000054b9: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000054bd: c5 f9 6f 15 ab 1c 00 00     	vmovdqa	7339(%rip), %xmm2
1000054c5: c4 e2 21 00 da              	vpshufb	%xmm2, %xmm11, %xmm3
1000054ca: c4 e2 29 00 e2              	vpshufb	%xmm2, %xmm10, %xmm4
1000054cf: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000054d3: c4 e3 61 02 c9 0c           	vpblendd	$12, %xmm1, %xmm3, %xmm1
1000054d9: c4 e2 09 00 d8              	vpshufb	%xmm0, %xmm14, %xmm3
1000054de: c4 e2 51 00 c0              	vpshufb	%xmm0, %xmm5, %xmm0
1000054e3: c5 f9 62 c3                 	vpunpckldq	%xmm3, %xmm0, %xmm0
1000054e7: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000054ed: c4 e2 49 00 da              	vpshufb	%xmm2, %xmm6, %xmm3
1000054f2: c4 e2 41 00 d2              	vpshufb	%xmm2, %xmm7, %xmm2
1000054f7: c5 e9 62 d3                 	vpunpckldq	%xmm3, %xmm2, %xmm2
1000054fb: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005501: c4 e3 6d 02 c0 c0           	vpblendd	$192, %ymm0, %ymm2, %ymm0
100005507: c4 63 75 02 c8 f0           	vpblendd	$240, %ymm0, %ymm1, %ymm9
10000550d: c5 f9 6f 05 6b 1c 00 00     	vmovdqa	7275(%rip), %xmm0
100005515: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000551a: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
10000551f: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005523: c5 f9 6f 15 65 1c 00 00     	vmovdqa	7269(%rip), %xmm2
10000552b: c4 e2 21 00 da              	vpshufb	%xmm2, %xmm11, %xmm3
100005530: c4 e2 29 00 e2              	vpshufb	%xmm2, %xmm10, %xmm4
100005535: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005539: c4 e3 61 02 c9 0c           	vpblendd	$12, %xmm1, %xmm3, %xmm1
10000553f: c4 e2 09 00 d8              	vpshufb	%xmm0, %xmm14, %xmm3
100005544: c4 e2 51 00 c0              	vpshufb	%xmm0, %xmm5, %xmm0
100005549: c5 f9 62 c3                 	vpunpckldq	%xmm3, %xmm0, %xmm0
10000554d: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005553: c4 e2 49 00 da              	vpshufb	%xmm2, %xmm6, %xmm3
100005558: c4 e2 41 00 d2              	vpshufb	%xmm2, %xmm7, %xmm2
10000555d: c5 e9 62 d3                 	vpunpckldq	%xmm3, %xmm2, %xmm2
100005561: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005567: c4 e3 6d 02 c0 c0           	vpblendd	$192, %ymm0, %ymm2, %ymm0
10000556d: c4 63 75 02 f8 f0           	vpblendd	$240, %ymm0, %ymm1, %ymm15
100005573: c5 f9 6f 0d 25 1c 00 00     	vmovdqa	7205(%rip), %xmm1
10000557b: c4 e2 11 00 d1              	vpshufb	%xmm1, %xmm13, %xmm2
100005580: c4 e2 19 00 d9              	vpshufb	%xmm1, %xmm12, %xmm3
100005585: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005589: c5 f9 6f 1d 1f 1c 00 00     	vmovdqa	7199(%rip), %xmm3
100005591: c4 e2 21 00 e3              	vpshufb	%xmm3, %xmm11, %xmm4
100005596: c4 e2 29 00 c3              	vpshufb	%xmm3, %xmm10, %xmm0
10000559b: c5 f9 62 c4                 	vpunpckldq	%xmm4, %xmm0, %xmm0
10000559f: c4 e3 79 02 c2 0c           	vpblendd	$12, %xmm2, %xmm0, %xmm0
1000055a5: c4 e2 09 00 d1              	vpshufb	%xmm1, %xmm14, %xmm2
1000055aa: c4 e2 51 00 c9              	vpshufb	%xmm1, %xmm5, %xmm1
1000055af: c5 f1 62 ca                 	vpunpckldq	%xmm2, %xmm1, %xmm1
1000055b3: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000055b9: c4 e2 49 00 d3              	vpshufb	%xmm3, %xmm6, %xmm2
1000055be: c4 e2 41 00 db              	vpshufb	%xmm3, %xmm7, %xmm3
1000055c3: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000055c7: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000055cd: c4 e3 6d 02 c9 c0           	vpblendd	$192, %ymm1, %ymm2, %ymm1
1000055d3: c4 e3 7d 02 c1 f0           	vpblendd	$240, %ymm1, %ymm0, %ymm0
1000055d9: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
1000055dd: c4 e2 3d 3c d1              	vpmaxsb	%ymm1, %ymm8, %ymm2
1000055e2: c4 e2 35 3c d9              	vpmaxsb	%ymm1, %ymm9, %ymm3
1000055e7: c4 e2 05 3c e1              	vpmaxsb	%ymm1, %ymm15, %ymm4
1000055ec: c4 e2 7d 3c c1              	vpmaxsb	%ymm1, %ymm0, %ymm0
1000055f1: c5 ed 60 cb                 	vpunpcklbw	%ymm3, %ymm2, %ymm1
1000055f5: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000055f9: c5 dd 60 d8                 	vpunpcklbw	%ymm0, %ymm4, %ymm3
1000055fd: c5 dd 68 c0                 	vpunpckhbw	%ymm0, %ymm4, %ymm0
100005601: c5 f5 61 e3                 	vpunpcklwd	%ymm3, %ymm1, %ymm4
100005605: c5 f5 69 cb                 	vpunpckhwd	%ymm3, %ymm1, %ymm1
100005609: c5 ed 61 d8                 	vpunpcklwd	%ymm0, %ymm2, %ymm3
10000560d: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100005611: c4 e3 5d 38 d1 01           	vinserti128	$1, %xmm1, %ymm4, %ymm2
100005617: c4 e3 65 38 e8 01           	vinserti128	$1, %xmm0, %ymm3, %ymm5
10000561d: c4 e3 5d 46 c9 31           	vperm2i128	$49, %ymm1, %ymm4, %ymm1
100005623: c4 e3 65 46 c0 31           	vperm2i128	$49, %ymm0, %ymm3, %ymm0
100005629: c5 fe 7f 4c 0f 40           	vmovdqu	%ymm1, 64(%rdi,%rcx)
10000562f: c5 fe 7f 44 0f 60           	vmovdqu	%ymm0, 96(%rdi,%rcx)
100005635: c5 fe 7f 6c 0f 20           	vmovdqu	%ymm5, 32(%rdi,%rcx)
10000563b: c5 fe 7f 14 0f              	vmovdqu	%ymm2, (%rdi,%rcx)
100005640: 4a 8d 34 86                 	leaq	(%rsi,%r8,4), %rsi
100005644: 4a 8d 3c 87                 	leaq	(%rdi,%r8,4), %rdi
100005648: 4d 39 d0                    	cmpq	%r10, %r8
10000564b: 0f 84 a1 f6 ff ff           	je	-2399 <__Z4ReLUPaS_j+0x92>
100005651: 41 c1 e0 02                 	shll	$2, %r8d
100005655: 89 d0                       	movl	%edx, %eax
100005657: 44 29 c0                    	subl	%r8d, %eax
10000565a: e9 47 f6 ff ff              	jmp	-2489 <__Z4ReLUPaS_j+0x46>
10000565f: 31 c9                       	xorl	%ecx, %ecx
100005661: 4d 85 c9                    	testq	%r9, %r9
100005664: 0f 85 8a f7 ff ff           	jne	-2166 <__Z4ReLUPaS_j+0x194>
10000566a: e9 bb f7 ff ff              	jmp	-2117 <__Z4ReLUPaS_j+0x1ca>
10000566f: 31 c9                       	xorl	%ecx, %ecx
100005671: 4d 85 c9                    	testq	%r9, %r9
100005674: 0f 85 88 fd ff ff           	jne	-632 <__Z4ReLUPaS_j+0x7a2>
10000567a: eb c4                       	jmp	-60 <__Z4ReLUPaS_j+0x9e0>
10000567c: 90                          	nop
10000567d: 90                          	nop
10000567e: 90                          	nop
10000567f: 90                          	nop

0000000100005680 __ZN11LineNetworkC2Ev:
100005680: 55                          	pushq	%rbp
100005681: 48 89 e5                    	movq	%rsp, %rbp
100005684: 41 56                       	pushq	%r14
100005686: 53                          	pushq	%rbx
100005687: 48 89 fb                    	movq	%rdi, %rbx
10000568a: e8 f1 f2 ff ff              	callq	-3343 <__ZN14ModelInterfaceC2Ev>
10000568f: 48 8d 05 6a 3a 00 00        	leaq	14954(%rip), %rax
100005696: 48 89 03                    	movq	%rax, (%rbx)
100005699: 48 89 df                    	movq	%rbx, %rdi
10000569c: be 00 00 08 00              	movl	$524288, %esi
1000056a1: e8 1a f4 ff ff              	callq	-3046 <__ZN14ModelInterface11init_bufferEj>
1000056a6: c7 43 20 00 08 16 03        	movl	$51775488, 32(%rbx)
1000056ad: c5 f8 28 05 0b 1b 00 00     	vmovaps	6923(%rip), %xmm0
1000056b5: c5 f8 11 43 08              	vmovups	%xmm0, 8(%rbx)
1000056ba: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
1000056c4: 48 89 43 18                 	movq	%rax, 24(%rbx)
1000056c8: 5b                          	popq	%rbx
1000056c9: 41 5e                       	popq	%r14
1000056cb: 5d                          	popq	%rbp
1000056cc: c3                          	retq
1000056cd: 49 89 c6                    	movq	%rax, %r14
1000056d0: 48 89 df                    	movq	%rbx, %rdi
1000056d3: e8 e8 f2 ff ff              	callq	-3352 <__ZN14ModelInterfaceD2Ev>
1000056d8: 4c 89 f7                    	movq	%r14, %rdi
1000056db: e8 22 17 00 00              	callq	5922 <dyld_stub_binder+0x100006e02>
1000056e0: 0f 0b                       	ud2
1000056e2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000056ec: 0f 1f 40 00                 	nopl	(%rax)

00000001000056f0 __ZN11LineNetworkC1Ev:
1000056f0: 55                          	pushq	%rbp
1000056f1: 48 89 e5                    	movq	%rsp, %rbp
1000056f4: 41 56                       	pushq	%r14
1000056f6: 53                          	pushq	%rbx
1000056f7: 48 89 fb                    	movq	%rdi, %rbx
1000056fa: e8 81 f2 ff ff              	callq	-3455 <__ZN14ModelInterfaceC2Ev>
1000056ff: 48 8d 05 fa 39 00 00        	leaq	14842(%rip), %rax
100005706: 48 89 03                    	movq	%rax, (%rbx)
100005709: 48 89 df                    	movq	%rbx, %rdi
10000570c: be 00 00 08 00              	movl	$524288, %esi
100005711: e8 aa f3 ff ff              	callq	-3158 <__ZN14ModelInterface11init_bufferEj>
100005716: c7 43 20 00 08 16 03        	movl	$51775488, 32(%rbx)
10000571d: c5 f8 28 05 9b 1a 00 00     	vmovaps	6811(%rip), %xmm0
100005725: c5 f8 11 43 08              	vmovups	%xmm0, 8(%rbx)
10000572a: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
100005734: 48 89 43 18                 	movq	%rax, 24(%rbx)
100005738: 5b                          	popq	%rbx
100005739: 41 5e                       	popq	%r14
10000573b: 5d                          	popq	%rbp
10000573c: c3                          	retq
10000573d: 49 89 c6                    	movq	%rax, %r14
100005740: 48 89 df                    	movq	%rbx, %rdi
100005743: e8 78 f2 ff ff              	callq	-3464 <__ZN14ModelInterfaceD2Ev>
100005748: 4c 89 f7                    	movq	%r14, %rdi
10000574b: e8 b2 16 00 00              	callq	5810 <dyld_stub_binder+0x100006e02>
100005750: 0f 0b                       	ud2
100005752: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000575c: 0f 1f 40 00                 	nopl	(%rax)

0000000100005760 __ZN11LineNetwork7forwardEv:
100005760: 55                          	pushq	%rbp
100005761: 48 89 e5                    	movq	%rsp, %rbp
100005764: 41 57                       	pushq	%r15
100005766: 41 56                       	pushq	%r14
100005768: 41 55                       	pushq	%r13
10000576a: 41 54                       	pushq	%r12
10000576c: 53                          	pushq	%rbx
10000576d: 48 83 ec 58                 	subq	$88, %rsp
100005771: 49 89 ff                    	movq	%rdi, %r15
100005774: e8 27 f3 ff ff              	callq	-3289 <__ZN14ModelInterface13output_bufferEv>
100005779: 49 89 c6                    	movq	%rax, %r14
10000577c: 4c 89 ff                    	movq	%r15, %rdi
10000577f: e8 0c f3 ff ff              	callq	-3316 <__ZN14ModelInterface12input_bufferEv>
100005784: 48 8d 15 f5 1b 00 00        	leaq	7157(%rip), %rdx
10000578b: 48 8d 0d 36 1c 00 00        	leaq	7222(%rip), %rcx
100005792: 4c 89 f7                    	movq	%r14, %rdi
100005795: 48 89 c6                    	movq	%rax, %rsi
100005798: 41 b8 37 00 00 00           	movl	$55, %r8d
10000579e: e8 ad 05 00 00              	callq	1453 <__ZN11LineNetwork7forwardEv+0x5f0>
1000057a3: 4c 89 ff                    	movq	%r15, %rdi
1000057a6: e8 a5 f4 ff ff              	callq	-2907 <__ZN14ModelInterface11swap_bufferEv>
1000057ab: 4c 89 ff                    	movq	%r15, %rdi
1000057ae: e8 ed f2 ff ff              	callq	-3347 <__ZN14ModelInterface13output_bufferEv>
1000057b3: 49 89 c6                    	movq	%rax, %r14
1000057b6: 4c 89 ff                    	movq	%r15, %rdi
1000057b9: e8 d2 f2 ff ff              	callq	-3374 <__ZN14ModelInterface12input_bufferEv>
1000057be: 4c 89 f7                    	movq	%r14, %rdi
1000057c1: 48 89 c6                    	movq	%rax, %rsi
1000057c4: ba 00 00 08 00              	movl	$524288, %edx
1000057c9: e8 92 f4 ff ff              	callq	-2926 <__Z4ReLUPaS_j>
1000057ce: 4c 89 ff                    	movq	%r15, %rdi
1000057d1: e8 7a f4 ff ff              	callq	-2950 <__ZN14ModelInterface11swap_bufferEv>
1000057d6: 4c 89 ff                    	movq	%r15, %rdi
1000057d9: e8 c2 f2 ff ff              	callq	-3390 <__ZN14ModelInterface13output_bufferEv>
1000057de: 49 89 c5                    	movq	%rax, %r13
1000057e1: 4c 89 7d 88                 	movq	%r15, -120(%rbp)
1000057e5: 4c 89 ff                    	movq	%r15, %rdi
1000057e8: e8 a3 f2 ff ff              	callq	-3421 <__ZN14ModelInterface12input_bufferEv>
1000057ed: 48 89 45 d0                 	movq	%rax, -48(%rbp)
1000057f1: 31 c0                       	xorl	%eax, %eax
1000057f3: eb 23                       	jmp	35 <__ZN11LineNetwork7forwardEv+0xb8>
1000057f5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000057ff: 90                          	nop
100005800: 48 8b 45 c8                 	movq	-56(%rbp), %rax
100005804: 48 ff c0                    	incq	%rax
100005807: 4c 8b 6d c0                 	movq	-64(%rbp), %r13
10000580b: 49 ff c5                    	incq	%r13
10000580e: 48 83 f8 08                 	cmpq	$8, %rax
100005812: 0f 84 02 01 00 00           	je	258 <__ZN11LineNetwork7forwardEv+0x1ba>
100005818: 48 89 45 c8                 	movq	%rax, -56(%rbp)
10000581c: 48 8d 04 c5 00 00 00 00     	leaq	(,%rax,8), %rax
100005824: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005828: 48 8d 05 a1 1b 00 00        	leaq	7073(%rip), %rax
10000582f: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005833: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005837: 48 8d 54 08 18              	leaq	24(%rax,%rcx), %rdx
10000583c: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100005840: 48 89 4d 90                 	movq	%rcx, -112(%rbp)
100005844: 48 8d 44 08 30              	leaq	48(%rax,%rcx), %rax
100005849: 48 89 45 a8                 	movq	%rax, -88(%rbp)
10000584d: 4c 89 6d c0                 	movq	%r13, -64(%rbp)
100005851: 4c 8b 7d d0                 	movq	-48(%rbp), %r15
100005855: 31 c0                       	xorl	%eax, %eax
100005857: eb 25                       	jmp	37 <__ZN11LineNetwork7forwardEv+0x11e>
100005859: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005860: 4c 8b 7d b0                 	movq	-80(%rbp), %r15
100005864: 49 81 c7 00 10 00 00        	addq	$4096, %r15
10000586b: 49 81 c5 00 04 00 00        	addq	$1024, %r13
100005872: 48 8b 45 b8                 	movq	-72(%rbp), %rax
100005876: 48 3d fd 00 00 00           	cmpq	$253, %rax
10000587c: 73 82                       	jae	-126 <__ZN11LineNetwork7forwardEv+0xa0>
10000587e: 48 83 c0 02                 	addq	$2, %rax
100005882: 48 89 45 b8                 	movq	%rax, -72(%rbp)
100005886: 4c 89 7d b0                 	movq	%r15, -80(%rbp)
10000588a: 31 db                       	xorl	%ebx, %ebx
10000588c: eb 18                       	jmp	24 <__ZN11LineNetwork7forwardEv+0x146>
10000588e: 66 90                       	nop
100005890: 41 88 44 9d 00              	movb	%al, (%r13,%rbx,4)
100005895: 48 83 c3 02                 	addq	$2, %rbx
100005899: 49 83 c7 10                 	addq	$16, %r15
10000589d: 48 81 fb fd 00 00 00        	cmpq	$253, %rbx
1000058a4: 73 ba                       	jae	-70 <__ZN11LineNetwork7forwardEv+0x100>
1000058a6: 4c 89 ff                    	movq	%r15, %rdi
1000058a9: 48 8b 75 98                 	movq	-104(%rbp), %rsi
1000058ad: e8 5e 12 00 00              	callq	4702 <__ZN11LineNetwork7forwardEv+0x13b0>
1000058b2: 41 89 c6                    	movl	%eax, %r14d
1000058b5: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
1000058bc: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
1000058c0: e8 4b 12 00 00              	callq	4683 <__ZN11LineNetwork7forwardEv+0x13b0>
1000058c5: 41 89 c4                    	movl	%eax, %r12d
1000058c8: 45 01 f4                    	addl	%r14d, %r12d
1000058cb: 49 8d bf 00 10 00 00        	leaq	4096(%r15), %rdi
1000058d2: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
1000058d6: e8 35 12 00 00              	callq	4661 <__ZN11LineNetwork7forwardEv+0x13b0>
1000058db: 44 01 e0                    	addl	%r12d, %eax
1000058de: 48 8d 0d 2b 1d 00 00        	leaq	7467(%rip), %rcx
1000058e5: 48 8b 55 90                 	movq	-112(%rbp), %rdx
1000058e9: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
1000058ed: 01 c1                       	addl	%eax, %ecx
1000058ef: 6b c9 37                    	imull	$55, %ecx, %ecx
1000058f2: 89 c8                       	movl	%ecx, %eax
1000058f4: c1 f8 1f                    	sarl	$31, %eax
1000058f7: c1 e8 12                    	shrl	$18, %eax
1000058fa: 01 c8                       	addl	%ecx, %eax
1000058fc: c1 f8 0e                    	sarl	$14, %eax
1000058ff: 3d 80 00 00 00              	cmpl	$128, %eax
100005904: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x1ab>
100005906: b8 7f 00 00 00              	movl	$127, %eax
10000590b: 83 f8 81                    	cmpl	$-127, %eax
10000590e: 7f 80                       	jg	-128 <__ZN11LineNetwork7forwardEv+0x130>
100005910: b8 81 00 00 00              	movl	$129, %eax
100005915: e9 76 ff ff ff              	jmp	-138 <__ZN11LineNetwork7forwardEv+0x130>
10000591a: 4c 8b 7d 88                 	movq	-120(%rbp), %r15
10000591e: 4c 89 ff                    	movq	%r15, %rdi
100005921: e8 2a f3 ff ff              	callq	-3286 <__ZN14ModelInterface11swap_bufferEv>
100005926: 4c 89 ff                    	movq	%r15, %rdi
100005929: e8 72 f1 ff ff              	callq	-3726 <__ZN14ModelInterface13output_bufferEv>
10000592e: 49 89 c6                    	movq	%rax, %r14
100005931: 4c 89 ff                    	movq	%r15, %rdi
100005934: e8 57 f1 ff ff              	callq	-3753 <__ZN14ModelInterface12input_bufferEv>
100005939: 4c 89 f7                    	movq	%r14, %rdi
10000593c: 48 89 c6                    	movq	%rax, %rsi
10000593f: ba 00 00 02 00              	movl	$131072, %edx
100005944: e8 17 f3 ff ff              	callq	-3305 <__Z4ReLUPaS_j>
100005949: 4c 89 ff                    	movq	%r15, %rdi
10000594c: e8 ff f2 ff ff              	callq	-3329 <__ZN14ModelInterface11swap_bufferEv>
100005951: 4c 89 ff                    	movq	%r15, %rdi
100005954: e8 47 f1 ff ff              	callq	-3769 <__ZN14ModelInterface13output_bufferEv>
100005959: 49 89 c5                    	movq	%rax, %r13
10000595c: 4c 89 ff                    	movq	%r15, %rdi
10000595f: e8 2c f1 ff ff              	callq	-3796 <__ZN14ModelInterface12input_bufferEv>
100005964: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100005968: 31 c0                       	xorl	%eax, %eax
10000596a: eb 1c                       	jmp	28 <__ZN11LineNetwork7forwardEv+0x228>
10000596c: 0f 1f 40 00                 	nopl	(%rax)
100005970: 48 8b 45 c8                 	movq	-56(%rbp), %rax
100005974: 48 ff c0                    	incq	%rax
100005977: 4c 8b 6d c0                 	movq	-64(%rbp), %r13
10000597b: 49 ff c5                    	incq	%r13
10000597e: 48 83 f8 10                 	cmpq	$16, %rax
100005982: 0f 84 ff 00 00 00           	je	255 <__ZN11LineNetwork7forwardEv+0x327>
100005988: 48 89 45 c8                 	movq	%rax, -56(%rbp)
10000598c: 48 8d 04 c5 00 00 00 00     	leaq	(,%rax,8), %rax
100005994: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005998: 48 8d 05 81 1c 00 00        	leaq	7297(%rip), %rax
10000599f: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
1000059a3: 48 89 55 98                 	movq	%rdx, -104(%rbp)
1000059a7: 48 8d 54 08 18              	leaq	24(%rax,%rcx), %rdx
1000059ac: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
1000059b0: 48 89 4d 90                 	movq	%rcx, -112(%rbp)
1000059b4: 48 8d 44 08 30              	leaq	48(%rax,%rcx), %rax
1000059b9: 48 89 45 a8                 	movq	%rax, -88(%rbp)
1000059bd: 4c 89 6d c0                 	movq	%r13, -64(%rbp)
1000059c1: 4c 8b 7d d0                 	movq	-48(%rbp), %r15
1000059c5: 31 c0                       	xorl	%eax, %eax
1000059c7: eb 23                       	jmp	35 <__ZN11LineNetwork7forwardEv+0x28c>
1000059c9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
1000059d0: 4c 8b 7d b0                 	movq	-80(%rbp), %r15
1000059d4: 49 81 c7 00 08 00 00        	addq	$2048, %r15
1000059db: 49 81 c5 00 04 00 00        	addq	$1024, %r13
1000059e2: 48 8b 45 b8                 	movq	-72(%rbp), %rax
1000059e6: 48 83 f8 7d                 	cmpq	$125, %rax
1000059ea: 73 84                       	jae	-124 <__ZN11LineNetwork7forwardEv+0x210>
1000059ec: 48 83 c0 02                 	addq	$2, %rax
1000059f0: 48 89 45 b8                 	movq	%rax, -72(%rbp)
1000059f4: 4c 89 7d b0                 	movq	%r15, -80(%rbp)
1000059f8: 31 db                       	xorl	%ebx, %ebx
1000059fa: eb 17                       	jmp	23 <__ZN11LineNetwork7forwardEv+0x2b3>
1000059fc: 0f 1f 40 00                 	nopl	(%rax)
100005a00: 41 88 44 dd 00              	movb	%al, (%r13,%rbx,8)
100005a05: 48 83 c3 02                 	addq	$2, %rbx
100005a09: 49 83 c7 10                 	addq	$16, %r15
100005a0d: 48 83 fb 7d                 	cmpq	$125, %rbx
100005a11: 73 bd                       	jae	-67 <__ZN11LineNetwork7forwardEv+0x270>
100005a13: 4c 89 ff                    	movq	%r15, %rdi
100005a16: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100005a1a: e8 f1 10 00 00              	callq	4337 <__ZN11LineNetwork7forwardEv+0x13b0>
100005a1f: 41 89 c6                    	movl	%eax, %r14d
100005a22: 49 8d bf 00 04 00 00        	leaq	1024(%r15), %rdi
100005a29: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005a2d: e8 de 10 00 00              	callq	4318 <__ZN11LineNetwork7forwardEv+0x13b0>
100005a32: 41 89 c4                    	movl	%eax, %r12d
100005a35: 45 01 f4                    	addl	%r14d, %r12d
100005a38: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
100005a3f: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100005a43: e8 c8 10 00 00              	callq	4296 <__ZN11LineNetwork7forwardEv+0x13b0>
100005a48: 44 01 e0                    	addl	%r12d, %eax
100005a4b: 48 8d 0d 4e 20 00 00        	leaq	8270(%rip), %rcx
100005a52: 48 8b 55 90                 	movq	-112(%rbp), %rdx
100005a56: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
100005a5a: 01 c1                       	addl	%eax, %ecx
100005a5c: 6b c9 39                    	imull	$57, %ecx, %ecx
100005a5f: 89 c8                       	movl	%ecx, %eax
100005a61: c1 f8 1f                    	sarl	$31, %eax
100005a64: c1 e8 12                    	shrl	$18, %eax
100005a67: 01 c8                       	addl	%ecx, %eax
100005a69: c1 f8 0e                    	sarl	$14, %eax
100005a6c: 3d 80 00 00 00              	cmpl	$128, %eax
100005a71: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x318>
100005a73: b8 7f 00 00 00              	movl	$127, %eax
100005a78: 83 f8 81                    	cmpl	$-127, %eax
100005a7b: 7f 83                       	jg	-125 <__ZN11LineNetwork7forwardEv+0x2a0>
100005a7d: b8 81 00 00 00              	movl	$129, %eax
100005a82: e9 79 ff ff ff              	jmp	-135 <__ZN11LineNetwork7forwardEv+0x2a0>
100005a87: 4c 8b 7d 88                 	movq	-120(%rbp), %r15
100005a8b: 4c 89 ff                    	movq	%r15, %rdi
100005a8e: e8 bd f1 ff ff              	callq	-3651 <__ZN14ModelInterface11swap_bufferEv>
100005a93: 4c 89 ff                    	movq	%r15, %rdi
100005a96: e8 05 f0 ff ff              	callq	-4091 <__ZN14ModelInterface13output_bufferEv>
100005a9b: 49 89 c6                    	movq	%rax, %r14
100005a9e: 4c 89 ff                    	movq	%r15, %rdi
100005aa1: e8 ea ef ff ff              	callq	-4118 <__ZN14ModelInterface12input_bufferEv>
100005aa6: 4c 89 f7                    	movq	%r14, %rdi
100005aa9: 48 89 c6                    	movq	%rax, %rsi
100005aac: ba 00 00 01 00              	movl	$65536, %edx
100005ab1: e8 aa f1 ff ff              	callq	-3670 <__Z4ReLUPaS_j>
100005ab6: 4c 89 ff                    	movq	%r15, %rdi
100005ab9: e8 92 f1 ff ff              	callq	-3694 <__ZN14ModelInterface11swap_bufferEv>
100005abe: 4c 89 ff                    	movq	%r15, %rdi
100005ac1: e8 da ef ff ff              	callq	-4134 <__ZN14ModelInterface13output_bufferEv>
100005ac6: 48 89 c3                    	movq	%rax, %rbx
100005ac9: 4c 89 ff                    	movq	%r15, %rdi
100005acc: e8 bf ef ff ff              	callq	-4161 <__ZN14ModelInterface12input_bufferEv>
100005ad1: 48 89 45 80                 	movq	%rax, -128(%rbp)
100005ad5: 31 c0                       	xorl	%eax, %eax
100005ad7: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x398>
100005ad9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005ae0: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100005ae4: 48 ff c0                    	incq	%rax
100005ae7: 48 8b 5d c8                 	movq	-56(%rbp), %rbx
100005aeb: 48 ff c3                    	incq	%rbx
100005aee: 48 83 f8 20                 	cmpq	$32, %rax
100005af2: 0f 84 17 01 00 00           	je	279 <__ZN11LineNetwork7forwardEv+0x4af>
100005af8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100005afc: 48 c1 e0 04                 	shlq	$4, %rax
100005b00: 48 8d 0c c0                 	leaq	(%rax,%rax,8), %rcx
100005b04: 48 8d 05 a5 1f 00 00        	leaq	8101(%rip), %rax
100005b0b: 48 8d 14 08                 	leaq	(%rax,%rcx), %rdx
100005b0f: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100005b13: 48 8d 54 08 30              	leaq	48(%rax,%rcx), %rdx
100005b18: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100005b1c: 48 89 4d 90                 	movq	%rcx, -112(%rbp)
100005b20: 48 8d 44 08 60              	leaq	96(%rax,%rcx), %rax
100005b25: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100005b29: 48 89 5d c8                 	movq	%rbx, -56(%rbp)
100005b2d: 4c 8b 7d 80                 	movq	-128(%rbp), %r15
100005b31: 31 c0                       	xorl	%eax, %eax
100005b33: eb 2b                       	jmp	43 <__ZN11LineNetwork7forwardEv+0x400>
100005b35: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005b3f: 90                          	nop
100005b40: 4c 8b 7d b8                 	movq	-72(%rbp), %r15
100005b44: 49 81 c7 00 08 00 00        	addq	$2048, %r15
100005b4b: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100005b4f: 48 81 c3 00 04 00 00        	addq	$1024, %rbx
100005b56: 48 8b 45 c0                 	movq	-64(%rbp), %rax
100005b5a: 48 83 f8 3d                 	cmpq	$61, %rax
100005b5e: 73 80                       	jae	-128 <__ZN11LineNetwork7forwardEv+0x380>
100005b60: 48 83 c0 02                 	addq	$2, %rax
100005b64: 48 89 45 c0                 	movq	%rax, -64(%rbp)
100005b68: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
100005b6c: 4c 89 7d b8                 	movq	%r15, -72(%rbp)
100005b70: 45 31 f6                    	xorl	%r14d, %r14d
100005b73: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x434>
100005b75: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005b7f: 90                          	nop
100005b80: 88 03                       	movb	%al, (%rbx)
100005b82: 49 83 c6 02                 	addq	$2, %r14
100005b86: 49 83 c7 20                 	addq	$32, %r15
100005b8a: 48 83 c3 20                 	addq	$32, %rbx
100005b8e: 49 83 fe 3d                 	cmpq	$61, %r14
100005b92: 73 ac                       	jae	-84 <__ZN11LineNetwork7forwardEv+0x3e0>
100005b94: 4c 89 ff                    	movq	%r15, %rdi
100005b97: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100005b9b: e8 f0 10 00 00              	callq	4336 <__ZN11LineNetwork7forwardEv+0x1530>
100005ba0: 41 89 c4                    	movl	%eax, %r12d
100005ba3: 49 8d bf 00 04 00 00        	leaq	1024(%r15), %rdi
100005baa: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005bae: e8 dd 10 00 00              	callq	4317 <__ZN11LineNetwork7forwardEv+0x1530>
100005bb3: 41 89 c5                    	movl	%eax, %r13d
100005bb6: 45 01 e5                    	addl	%r12d, %r13d
100005bb9: 49 8d bf 00 08 00 00        	leaq	2048(%r15), %rdi
100005bc0: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100005bc4: e8 c7 10 00 00              	callq	4295 <__ZN11LineNetwork7forwardEv+0x1530>
100005bc9: 44 01 e8                    	addl	%r13d, %eax
100005bcc: 48 8d 0d dd 30 00 00        	leaq	12509(%rip), %rcx
100005bd3: 48 8b 55 90                 	movq	-112(%rbp), %rdx
100005bd7: 0f be 0c 0a                 	movsbl	(%rdx,%rcx), %ecx
100005bdb: 01 c1                       	addl	%eax, %ecx
100005bdd: c1 e1 04                    	shll	$4, %ecx
100005be0: 8d 0c 49                    	leal	(%rcx,%rcx,2), %ecx
100005be3: 89 c8                       	movl	%ecx, %eax
100005be5: c1 f8 1f                    	sarl	$31, %eax
100005be8: c1 e8 12                    	shrl	$18, %eax
100005beb: 01 c8                       	addl	%ecx, %eax
100005bed: c1 f8 0e                    	sarl	$14, %eax
100005bf0: 3d 80 00 00 00              	cmpl	$128, %eax
100005bf5: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x49c>
100005bf7: b8 7f 00 00 00              	movl	$127, %eax
100005bfc: 83 f8 81                    	cmpl	$-127, %eax
100005bff: 0f 8f 7b ff ff ff           	jg	-133 <__ZN11LineNetwork7forwardEv+0x420>
100005c05: b8 81 00 00 00              	movl	$129, %eax
100005c0a: e9 71 ff ff ff              	jmp	-143 <__ZN11LineNetwork7forwardEv+0x420>
100005c0f: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100005c13: 48 89 df                    	movq	%rbx, %rdi
100005c16: e8 35 f0 ff ff              	callq	-4043 <__ZN14ModelInterface11swap_bufferEv>
100005c1b: 48 89 df                    	movq	%rbx, %rdi
100005c1e: e8 7d ee ff ff              	callq	-4483 <__ZN14ModelInterface13output_bufferEv>
100005c23: 49 89 c6                    	movq	%rax, %r14
100005c26: 48 89 df                    	movq	%rbx, %rdi
100005c29: e8 62 ee ff ff              	callq	-4510 <__ZN14ModelInterface12input_bufferEv>
100005c2e: 4c 89 f7                    	movq	%r14, %rdi
100005c31: 48 89 c6                    	movq	%rax, %rsi
100005c34: ba 00 80 00 00              	movl	$32768, %edx
100005c39: e8 22 f0 ff ff              	callq	-4062 <__Z4ReLUPaS_j>
100005c3e: 48 89 df                    	movq	%rbx, %rdi
100005c41: e8 0a f0 ff ff              	callq	-4086 <__ZN14ModelInterface11swap_bufferEv>
100005c46: 48 89 df                    	movq	%rbx, %rdi
100005c49: e8 52 ee ff ff              	callq	-4526 <__ZN14ModelInterface13output_bufferEv>
100005c4e: 49 89 c6                    	movq	%rax, %r14
100005c51: 48 89 df                    	movq	%rbx, %rdi
100005c54: e8 37 ee ff ff              	callq	-4553 <__ZN14ModelInterface12input_bufferEv>
100005c59: 48 83 c0 07                 	addq	$7, %rax
100005c5d: 45 31 c0                    	xorl	%r8d, %r8d
100005c60: c5 f9 6f 05 68 15 00 00     	vmovdqa	5480(%rip), %xmm0
100005c68: 41 b9 7f 00 00 00           	movl	$127, %r9d
100005c6e: 41 ba 81 00 00 00           	movl	$129, %r10d
100005c74: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005c7e: 66 90                       	nop
100005c80: 48 89 c7                    	movq	%rax, %rdi
100005c83: 31 c9                       	xorl	%ecx, %ecx
100005c85: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005c8f: 90                          	nop
100005c90: c4 e2 79 21 4f f9           	vpmovsxbd	-7(%rdi), %xmm1
100005c96: c4 e2 71 40 c8              	vpmulld	%xmm0, %xmm1, %xmm1
100005c9b: 0f be 57 fd                 	movsbl	-3(%rdi), %edx
100005c9f: 44 6b da 4a                 	imull	$74, %edx, %r11d
100005ca3: 0f be 77 fe                 	movsbl	-2(%rdi), %esi
100005ca7: c1 e6 06                    	shll	$6, %esi
100005caa: 0f be 5f ff                 	movsbl	-1(%rdi), %ebx
100005cae: 6b db 3c                    	imull	$60, %ebx, %ebx
100005cb1: 0f be 17                    	movsbl	(%rdi), %edx
100005cb4: 44 6b fa f3                 	imull	$-13, %edx, %r15d
100005cb8: c5 f9 70 d1 4e              	vpshufd	$78, %xmm1, %xmm2
100005cbd: c5 f1 fe ca                 	vpaddd	%xmm2, %xmm1, %xmm1
100005cc1: c5 f9 70 d1 e5              	vpshufd	$229, %xmm1, %xmm2
100005cc6: c5 f1 fe ca                 	vpaddd	%xmm2, %xmm1, %xmm1
100005cca: c5 f9 7e ca                 	vmovd	%xmm1, %edx
100005cce: 44 01 da                    	addl	%r11d, %edx
100005cd1: 01 da                       	addl	%ebx, %edx
100005cd3: 44 01 fa                    	addl	%r15d, %edx
100005cd6: 01 f2                       	addl	%esi, %edx
100005cd8: c1 e2 05                    	shll	$5, %edx
100005cdb: 89 d6                       	movl	%edx, %esi
100005cdd: 83 c6 20                    	addl	$32, %esi
100005ce0: c1 fe 1f                    	sarl	$31, %esi
100005ce3: c1 ee 12                    	shrl	$18, %esi
100005ce6: 8d 14 32                    	leal	(%rdx,%rsi), %edx
100005ce9: 83 c2 20                    	addl	$32, %edx
100005cec: c1 fa 0e                    	sarl	$14, %edx
100005cef: 81 fa 80 00 00 00           	cmpl	$128, %edx
100005cf5: 41 0f 4d d1                 	cmovgel	%r9d, %edx
100005cf9: 83 fa 81                    	cmpl	$-127, %edx
100005cfc: 41 0f 4e d2                 	cmovlel	%r10d, %edx
100005d00: 41 88 14 0e                 	movb	%dl, (%r14,%rcx)
100005d04: 48 ff c1                    	incq	%rcx
100005d07: 48 83 c7 20                 	addq	$32, %rdi
100005d0b: 48 83 f9 20                 	cmpq	$32, %rcx
100005d0f: 0f 85 7b ff ff ff           	jne	-133 <__ZN11LineNetwork7forwardEv+0x530>
100005d15: 49 ff c0                    	incq	%r8
100005d18: 49 83 c6 20                 	addq	$32, %r14
100005d1c: 48 05 00 04 00 00           	addq	$1024, %rax
100005d22: 49 83 f8 20                 	cmpq	$32, %r8
100005d26: 0f 85 54 ff ff ff           	jne	-172 <__ZN11LineNetwork7forwardEv+0x520>
100005d2c: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100005d30: 48 89 df                    	movq	%rbx, %rdi
100005d33: e8 18 ef ff ff              	callq	-4328 <__ZN14ModelInterface11swap_bufferEv>
100005d38: 48 89 df                    	movq	%rbx, %rdi
100005d3b: 48 83 c4 58                 	addq	$88, %rsp
100005d3f: 5b                          	popq	%rbx
100005d40: 41 5c                       	popq	%r12
100005d42: 41 5d                       	popq	%r13
100005d44: 41 5e                       	popq	%r14
100005d46: 41 5f                       	popq	%r15
100005d48: 5d                          	popq	%rbp
100005d49: e9 02 ef ff ff              	jmp	-4350 <__ZN14ModelInterface11swap_bufferEv>
100005d4e: 66 90                       	nop
100005d50: 55                          	pushq	%rbp
100005d51: 48 89 e5                    	movq	%rsp, %rbp
100005d54: 41 57                       	pushq	%r15
100005d56: 41 56                       	pushq	%r14
100005d58: 41 55                       	pushq	%r13
100005d5a: 41 54                       	pushq	%r12
100005d5c: 53                          	pushq	%rbx
100005d5d: 48 83 e4 e0                 	andq	$-32, %rsp
100005d61: 48 81 ec e0 02 00 00        	subq	$736, %rsp
100005d68: 48 89 4c 24 50              	movq	%rcx, 80(%rsp)
100005d6d: 48 89 54 24 48              	movq	%rdx, 72(%rsp)
100005d72: 49 89 ff                    	movq	%rdi, %r15
100005d75: c4 c1 79 6e c0              	vmovd	%r8d, %xmm0
100005d7a: c4 e2 7d 58 c8              	vpbroadcastd	%xmm0, %ymm1
100005d7f: 48 8d 86 01 04 00 00        	leaq	1025(%rsi), %rax
100005d86: 48 89 44 24 40              	movq	%rax, 64(%rsp)
100005d8b: 48 8d 86 02 04 00 00        	leaq	1026(%rsi), %rax
100005d92: 48 89 44 24 38              	movq	%rax, 56(%rsp)
100005d97: 45 31 c9                    	xorl	%r9d, %r9d
100005d9a: c5 fd 6f 15 5e 15 00 00     	vmovdqa	5470(%rip), %ymm2
100005da2: 44 89 44 24 14              	movl	%r8d, 20(%rsp)
100005da7: 48 89 74 24 58              	movq	%rsi, 88(%rsp)
100005dac: c5 fd 7f 8c 24 60 02 00 00  	vmovdqa	%ymm1, 608(%rsp)
100005db5: eb 19                       	jmp	25 <__ZN11LineNetwork7forwardEv+0x670>
100005db7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100005dc0: 49 ff c1                    	incq	%r9
100005dc3: 48 ff c7                    	incq	%rdi
100005dc6: 49 83 f9 08                 	cmpq	$8, %r9
100005dca: 0f 84 f2 0c 00 00           	je	3314 <__ZN11LineNetwork7forwardEv+0x1362>
100005dd0: 49 8d 81 f1 07 00 00        	leaq	2033(%r9), %rax
100005dd7: 48 89 84 24 88 00 00 00     	movq	%rax, 136(%rsp)
100005ddf: 4b 8d 04 c9                 	leaq	(%r9,%r9,8), %rax
100005de3: 48 8b 54 24 48              	movq	72(%rsp), %rdx
100005de8: 48 8d 0c 02                 	leaq	(%rdx,%rax), %rcx
100005dec: 48 83 c1 09                 	addq	$9, %rcx
100005df0: 48 89 8c 24 80 00 00 00     	movq	%rcx, 128(%rsp)
100005df8: 48 8b 4c 24 50              	movq	80(%rsp), %rcx
100005dfd: 48 8d 5c 01 01              	leaq	1(%rcx,%rax), %rbx
100005e02: 48 89 5c 24 78              	movq	%rbx, 120(%rsp)
100005e07: 4c 8d 14 02                 	leaq	(%rdx,%rax), %r10
100005e0b: 4c 8d 1c 01                 	leaq	(%rcx,%rax), %r11
100005e0f: 48 8d 44 02 08              	leaq	8(%rdx,%rax), %rax
100005e14: 48 89 44 24 70              	movq	%rax, 112(%rsp)
100005e19: c4 c1 f9 6e c1              	vmovq	%r9, %xmm0
100005e1e: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005e23: 41 be 00 00 00 00           	movl	$0, %r14d
100005e29: 48 8b 44 24 38              	movq	56(%rsp), %rax
100005e2e: 48 89 44 24 30              	movq	%rax, 48(%rsp)
100005e33: 48 8b 44 24 40              	movq	64(%rsp), %rax
100005e38: 31 c9                       	xorl	%ecx, %ecx
100005e3a: 31 d2                       	xorl	%edx, %edx
100005e3c: 48 89 54 24 08              	movq	%rdx, 8(%rsp)
100005e41: 4c 89 4c 24 68              	movq	%r9, 104(%rsp)
100005e46: 48 89 7c 24 60              	movq	%rdi, 96(%rsp)
100005e4b: 4c 89 54 24 20              	movq	%r10, 32(%rsp)
100005e50: 4c 89 5c 24 18              	movq	%r11, 24(%rsp)
100005e55: c5 fd 7f 84 24 80 02 00 00  	vmovdqa	%ymm0, 640(%rsp)
100005e5e: eb 38                       	jmp	56 <__ZN11LineNetwork7forwardEv+0x738>
100005e60: 48 8b 8c 24 90 00 00 00     	movq	144(%rsp), %rcx
100005e68: 48 ff c1                    	incq	%rcx
100005e6b: 48 8b 44 24 28              	movq	40(%rsp), %rax
100005e70: 48 05 00 04 00 00           	addq	$1024, %rax
100005e76: 48 81 44 24 30 00 04 00 00  	addq	$1024, 48(%rsp)
100005e7f: 49 81 c6 00 01 00 00        	addq	$256, %r14
100005e86: 48 81 7c 24 08 fd 01 00 00  	cmpq	$509, 8(%rsp)
100005e8f: 4d 89 e9                    	movq	%r13, %r9
100005e92: 0f 83 28 ff ff ff           	jae	-216 <__ZN11LineNetwork7forwardEv+0x660>
100005e98: 48 89 44 24 28              	movq	%rax, 40(%rsp)
100005e9d: 4c 89 b4 24 98 00 00 00     	movq	%r14, 152(%rsp)
100005ea5: 48 89 cb                    	movq	%rcx, %rbx
100005ea8: 48 c1 e3 0b                 	shlq	$11, %rbx
100005eac: 4d 89 cd                    	movq	%r9, %r13
100005eaf: 49 8d 04 19                 	leaq	(%r9,%rbx), %rax
100005eb3: 4c 01 f8                    	addq	%r15, %rax
100005eb6: 48 03 9c 24 88 00 00 00     	addq	136(%rsp), %rbx
100005ebe: 4c 01 fb                    	addq	%r15, %rbx
100005ec1: 48 89 ca                    	movq	%rcx, %rdx
100005ec4: 48 c1 e2 0a                 	shlq	$10, %rdx
100005ec8: 4c 8d 0c 16                 	leaq	(%rsi,%rdx), %r9
100005ecc: 49 81 c1 ff 05 00 00        	addq	$1535, %r9
100005ed3: 48 01 f2                    	addq	%rsi, %rdx
100005ed6: 4c 39 c8                    	cmpq	%r9, %rax
100005ed9: 41 0f 92 c4                 	setb	%r12b
100005edd: 48 39 da                    	cmpq	%rbx, %rdx
100005ee0: 41 0f 92 c2                 	setb	%r10b
100005ee4: 48 3b 84 24 80 00 00 00     	cmpq	128(%rsp), %rax
100005eec: 41 0f 92 c6                 	setb	%r14b
100005ef0: 48 39 5c 24 70              	cmpq	%rbx, 112(%rsp)
100005ef5: 4c 89 da                    	movq	%r11, %rdx
100005ef8: 41 0f 92 c3                 	setb	%r11b
100005efc: 48 3b 44 24 78              	cmpq	120(%rsp), %rax
100005f01: 0f 92 c0                    	setb	%al
100005f04: 48 39 da                    	cmpq	%rbx, %rdx
100005f07: 41 0f 92 c1                 	setb	%r9b
100005f0b: 45 84 d4                    	testb	%r10b, %r12b
100005f0e: 48 89 8c 24 90 00 00 00     	movq	%rcx, 144(%rsp)
100005f16: 0f 85 84 0a 00 00           	jne	2692 <__ZN11LineNetwork7forwardEv+0x1240>
100005f1c: 45 20 de                    	andb	%r11b, %r14b
100005f1f: 0f 85 7b 0a 00 00           	jne	2683 <__ZN11LineNetwork7forwardEv+0x1240>
100005f25: ba 00 00 00 00              	movl	$0, %edx
100005f2a: 44 20 c8                    	andb	%r9b, %al
100005f2d: 0f 85 6f 0a 00 00           	jne	2671 <__ZN11LineNetwork7forwardEv+0x1242>
100005f33: 48 8b 44 24 08              	movq	8(%rsp), %rax
100005f38: 48 c1 e0 07                 	shlq	$7, %rax
100005f3c: c4 e1 f9 6e c0              	vmovq	%rax, %xmm0
100005f41: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005f46: c5 fd 7f 84 24 a0 02 00 00  	vmovdqa	%ymm0, 672(%rsp)
100005f4f: 45 31 db                    	xorl	%r11d, %r11d
100005f52: c5 fc 28 05 86 13 00 00     	vmovaps	4998(%rip), %ymm0
100005f5a: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100005f63: c5 fc 28 05 55 13 00 00     	vmovaps	4949(%rip), %ymm0
100005f6b: c5 fc 29 84 24 20 02 00 00  	vmovaps	%ymm0, 544(%rsp)
100005f74: c5 fc 28 05 24 13 00 00     	vmovaps	4900(%rip), %ymm0
100005f7c: c5 fc 29 84 24 00 02 00 00  	vmovaps	%ymm0, 512(%rsp)
100005f85: c5 fc 28 05 f3 12 00 00     	vmovaps	4851(%rip), %ymm0
100005f8d: c5 fc 29 84 24 e0 01 00 00  	vmovaps	%ymm0, 480(%rsp)
100005f96: c5 fc 28 05 c2 12 00 00     	vmovaps	4802(%rip), %ymm0
100005f9e: c5 fc 29 84 24 c0 01 00 00  	vmovaps	%ymm0, 448(%rsp)
100005fa7: c5 fc 28 05 91 12 00 00     	vmovaps	4753(%rip), %ymm0
100005faf: c5 fc 29 84 24 a0 01 00 00  	vmovaps	%ymm0, 416(%rsp)
100005fb8: c5 fc 28 05 60 12 00 00     	vmovaps	4704(%rip), %ymm0
100005fc0: c5 fc 29 84 24 80 01 00 00  	vmovaps	%ymm0, 384(%rsp)
100005fc9: c5 fc 28 05 2f 12 00 00     	vmovaps	4655(%rip), %ymm0
100005fd1: c5 fc 29 84 24 60 01 00 00  	vmovaps	%ymm0, 352(%rsp)
100005fda: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100005fe0: 48 8b 4c 24 28              	movq	40(%rsp), %rcx
100005fe5: c4 a1 7e 6f 84 59 1f fc ff ff       	vmovdqu	-993(%rcx,%r11,2), %ymm0
100005fef: c4 e2 7d 00 c2              	vpshufb	%ymm2, %ymm0, %ymm0
100005ff4: c4 a1 7e 6f 8c 59 ff fb ff ff       	vmovdqu	-1025(%rcx,%r11,2), %ymm1
100005ffe: c4 21 7e 6f 84 59 00 fc ff ff       	vmovdqu	-1024(%rcx,%r11,2), %ymm8
100006008: c5 7d 6f 1d 10 13 00 00     	vmovdqa	4880(%rip), %ymm11
100006010: c4 c2 75 00 cb              	vpshufb	%ymm11, %ymm1, %ymm1
100006015: c4 e3 75 02 c0 cc           	vpblendd	$204, %ymm0, %ymm1, %ymm0
10000601b: c4 e3 fd 00 c8 d8           	vpermq	$216, %ymm0, %ymm1
100006021: c4 a1 7a 6f 94 59 0f fc ff ff       	vmovdqu	-1009(%rcx,%r11,2), %xmm2
10000602b: c5 f9 6f 1d ad 11 00 00     	vmovdqa	4525(%rip), %xmm3
100006033: c4 e2 69 00 d3              	vpshufb	%xmm3, %xmm2, %xmm2
100006038: c5 79 6f e3                 	vmovdqa	%xmm3, %xmm12
10000603c: c4 62 7d 21 ca              	vpmovsxbd	%xmm2, %ymm9
100006041: c4 63 fd 00 d0 db           	vpermq	$219, %ymm0, %ymm10
100006047: 48 8b 44 24 20              	movq	32(%rsp), %rax
10000604c: c4 e2 79 78 00              	vpbroadcastb	(%rax), %xmm0
100006051: c4 e2 7d 21 d0              	vpmovsxbd	%xmm0, %ymm2
100006056: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
10000605b: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100006064: c4 62 7d 21 c9              	vpmovsxbd	%xmm1, %ymm9
100006069: c4 42 7d 21 d2              	vpmovsxbd	%xmm10, %ymm10
10000606e: c4 21 7e 6f ac 59 20 fc ff ff       	vmovdqu	-992(%rcx,%r11,2), %ymm13
100006078: c4 62 15 00 3d 7f 12 00 00  	vpshufb	4735(%rip), %ymm13, %ymm15
100006081: c4 c2 3d 00 fb              	vpshufb	%ymm11, %ymm8, %ymm7
100006086: c4 c3 45 02 ff cc           	vpblendd	$204, %ymm15, %ymm7, %ymm7
10000608c: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
100006092: c4 63 fd 00 ff d8           	vpermq	$216, %ymm7, %ymm15
100006098: c5 fd 6f 05 a0 12 00 00     	vmovdqa	4768(%rip), %ymm0
1000060a0: c4 62 15 00 e8              	vpshufb	%ymm0, %ymm13, %ymm13
1000060a5: c5 fd 6f 05 b3 12 00 00     	vmovdqa	4787(%rip), %ymm0
1000060ad: c4 62 3d 00 c0              	vpshufb	%ymm0, %ymm8, %ymm8
1000060b2: c4 c3 3d 02 f5 cc           	vpblendd	$204, %ymm13, %ymm8, %ymm6
1000060b8: c4 e3 fd 00 ee d8           	vpermq	$216, %ymm6, %ymm5
1000060be: c4 e2 7d 21 e1              	vpmovsxbd	%xmm1, %ymm4
1000060c3: c4 c2 7d 21 df              	vpmovsxbd	%xmm15, %ymm3
1000060c8: c4 e3 fd 00 cf db           	vpermq	$219, %ymm7, %ymm1
1000060ce: c4 62 7d 21 e9              	vpmovsxbd	%xmm1, %ymm13
1000060d3: c4 43 7d 39 ff 01           	vextracti128	$1, %ymm15, %xmm15
1000060d9: c4 42 6d 40 c2              	vpmulld	%ymm10, %ymm2, %ymm8
1000060de: c4 21 7a 6f 94 59 10 fc ff ff       	vmovdqu	-1008(%rcx,%r11,2), %xmm10
1000060e8: c4 c2 29 00 fc              	vpshufb	%xmm12, %xmm10, %xmm7
1000060ed: c4 62 79 78 70 01           	vpbroadcastb	1(%rax), %xmm14
1000060f3: c4 c2 6d 40 c1              	vpmulld	%ymm9, %ymm2, %ymm0
1000060f8: c5 fd 7f 84 24 a0 00 00 00  	vmovdqa	%ymm0, 160(%rsp)
100006101: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
100006106: c4 42 7d 21 f6              	vpmovsxbd	%xmm14, %ymm14
10000610b: c4 e2 0d 40 ff              	vpmulld	%ymm7, %ymm14, %ymm7
100006110: c4 42 0d 40 ed              	vpmulld	%ymm13, %ymm14, %ymm13
100006115: c4 42 7d 21 e7              	vpmovsxbd	%xmm15, %ymm12
10000611a: c4 c3 7d 39 ef 01           	vextracti128	$1, %ymm5, %xmm15
100006120: c4 e3 fd 00 f6 db           	vpermq	$219, %ymm6, %ymm6
100006126: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
10000612b: c4 62 0d 40 cb              	vpmulld	%ymm3, %ymm14, %ymm9
100006130: c4 e2 7d 21 dd              	vpmovsxbd	%xmm5, %ymm3
100006135: c5 f9 6f 05 b3 10 00 00     	vmovdqa	4275(%rip), %xmm0
10000613d: c4 e2 29 00 e8              	vpshufb	%xmm0, %xmm10, %xmm5
100006142: c4 e2 79 78 40 02           	vpbroadcastb	2(%rax), %xmm0
100006148: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
10000614d: c4 c2 7d 21 cf              	vpmovsxbd	%xmm15, %ymm1
100006152: c4 62 7d 40 fb              	vpmulld	%ymm3, %ymm0, %ymm15
100006157: c4 62 7d 40 d6              	vpmulld	%ymm6, %ymm0, %ymm10
10000615c: c4 e2 6d 40 d4              	vpmulld	%ymm4, %ymm2, %ymm2
100006161: c5 fd 7f 94 24 40 01 00 00  	vmovdqa	%ymm2, 320(%rsp)
10000616a: c4 e2 7d 21 d5              	vpmovsxbd	%xmm5, %ymm2
10000616f: c4 e2 7d 40 d2              	vpmulld	%ymm2, %ymm0, %ymm2
100006174: c4 a1 7e 6f 9c 59 ff fd ff ff       	vmovdqu	-513(%rcx,%r11,2), %ymm3
10000617e: c4 c2 0d 40 e4              	vpmulld	%ymm12, %ymm14, %ymm4
100006183: c5 fd 7f a4 24 00 01 00 00  	vmovdqa	%ymm4, 256(%rsp)
10000618c: c4 a1 7e 6f a4 59 1f fe ff ff       	vmovdqu	-481(%rcx,%r11,2), %ymm4
100006196: c4 e2 5d 00 25 61 11 00 00  	vpshufb	4449(%rip), %ymm4, %ymm4
10000619f: c4 c2 65 00 db              	vpshufb	%ymm11, %ymm3, %ymm3
1000061a4: c4 e3 65 02 dc cc           	vpblendd	$204, %ymm4, %ymm3, %ymm3
1000061aa: c4 e2 7d 40 c1              	vpmulld	%ymm1, %ymm0, %ymm0
1000061af: c5 fd 7f 84 24 20 01 00 00  	vmovdqa	%ymm0, 288(%rsp)
1000061b8: c4 e3 fd 00 c3 d8           	vpermq	$216, %ymm3, %ymm0
1000061be: c4 e2 7d 21 c8              	vpmovsxbd	%xmm0, %ymm1
1000061c3: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
1000061c9: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000061ce: c4 e3 7d 39 c0 01           	vextracti128	$1, %ymm0, %xmm0
1000061d4: c5 c5 fe a4 24 c0 00 00 00  	vpaddd	192(%rsp), %ymm7, %ymm4
1000061dd: c4 a1 7a 6f ac 59 0f fe ff ff       	vmovdqu	-497(%rcx,%r11,2), %xmm5
1000061e7: c5 79 6f 35 f1 0f 00 00     	vmovdqa	4081(%rip), %xmm14
1000061ef: c4 c2 51 00 ee              	vpshufb	%xmm14, %xmm5, %xmm5
1000061f4: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
1000061f9: c4 e2 79 78 70 03           	vpbroadcastb	3(%rax), %xmm6
1000061ff: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100006204: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
100006209: c4 e2 4d 40 c0              	vpmulld	%ymm0, %ymm6, %ymm0
10000620e: c5 fd 7f 84 24 c0 00 00 00  	vmovdqa	%ymm0, 192(%rsp)
100006217: c4 e2 4d 40 db              	vpmulld	%ymm3, %ymm6, %ymm3
10000621c: c4 41 3d fe ed              	vpaddd	%ymm13, %ymm8, %ymm13
100006221: c4 e2 4d 40 c9              	vpmulld	%ymm1, %ymm6, %ymm1
100006226: c4 e2 4d 40 c5              	vpmulld	%ymm5, %ymm6, %ymm0
10000622b: c5 ed fe c0                 	vpaddd	%ymm0, %ymm2, %ymm0
10000622f: c5 35 fe 84 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm9, %ymm8
100006238: c5 dd fe c0                 	vpaddd	%ymm0, %ymm4, %ymm0
10000623c: c5 fd 7f 84 24 e0 00 00 00  	vmovdqa	%ymm0, 224(%rsp)
100006245: c4 a1 7e 6f a4 59 00 fe ff ff       	vmovdqu	-512(%rcx,%r11,2), %ymm4
10000624f: c4 a1 7e 6f ac 59 20 fe ff ff       	vmovdqu	-480(%rcx,%r11,2), %ymm5
100006259: c4 e2 55 00 35 9e 10 00 00  	vpshufb	4254(%rip), %ymm5, %ymm6
100006262: c4 c2 5d 00 fb              	vpshufb	%ymm11, %ymm4, %ymm7
100006267: c5 2d fe d3                 	vpaddd	%ymm3, %ymm10, %ymm10
10000626b: c4 e3 45 02 de cc           	vpblendd	$204, %ymm6, %ymm7, %ymm3
100006271: c4 e3 fd 00 f3 d8           	vpermq	$216, %ymm3, %ymm6
100006277: c4 e3 7d 39 f7 01           	vextracti128	$1, %ymm6, %xmm7
10000627d: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
100006282: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
100006288: c5 05 fe c9                 	vpaddd	%ymm1, %ymm15, %ymm9
10000628c: c4 e2 7d 21 cb              	vpmovsxbd	%xmm3, %ymm1
100006291: c4 e2 7d 21 de              	vpmovsxbd	%xmm6, %ymm3
100006296: c4 a1 7a 6f b4 59 10 fe ff ff       	vmovdqu	-496(%rcx,%r11,2), %xmm6
1000062a0: c4 e2 79 78 40 04           	vpbroadcastb	4(%rax), %xmm0
1000062a6: c4 c2 49 00 d6              	vpshufb	%xmm14, %xmm6, %xmm2
1000062ab: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000062b0: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
1000062b5: c4 e2 7d 40 db              	vpmulld	%ymm3, %ymm0, %ymm3
1000062ba: c4 62 7d 40 e1              	vpmulld	%ymm1, %ymm0, %ymm12
1000062bf: c4 e2 7d 40 cf              	vpmulld	%ymm7, %ymm0, %ymm1
1000062c4: c5 fd 7f 8c 24 a0 00 00 00  	vmovdqa	%ymm1, 160(%rsp)
1000062cd: c4 e2 7d 40 c2              	vpmulld	%ymm2, %ymm0, %ymm0
1000062d2: c4 e2 55 00 15 65 10 00 00  	vpshufb	4197(%rip), %ymm5, %ymm2
1000062db: c4 e2 5d 00 25 7c 10 00 00  	vpshufb	4220(%rip), %ymm4, %ymm4
1000062e4: c4 e3 5d 02 d2 cc           	vpblendd	$204, %ymm2, %ymm4, %ymm2
1000062ea: c4 e2 79 78 60 05           	vpbroadcastb	5(%rax), %xmm4
1000062f0: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000062f5: c4 e3 fd 00 ea db           	vpermq	$219, %ymm2, %ymm5
1000062fb: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006300: c4 e2 5d 40 ed              	vpmulld	%ymm5, %ymm4, %ymm5
100006305: c5 9d fe ed                 	vpaddd	%ymm5, %ymm12, %ymm5
100006309: c4 e3 fd 00 d2 d8           	vpermq	$216, %ymm2, %ymm2
10000630f: c4 e2 7d 21 fa              	vpmovsxbd	%xmm2, %ymm7
100006314: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000631a: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000631f: c4 e2 49 00 35 c8 0e 00 00  	vpshufb	3784(%rip), %xmm6, %xmm6
100006328: c4 e2 5d 40 ff              	vpmulld	%ymm7, %ymm4, %ymm7
10000632d: c4 62 5d 40 fa              	vpmulld	%ymm2, %ymm4, %ymm15
100006332: c4 e2 7d 21 d6              	vpmovsxbd	%xmm6, %ymm2
100006337: c4 e2 5d 40 d2              	vpmulld	%ymm2, %ymm4, %ymm2
10000633c: c5 fd fe c2                 	vpaddd	%ymm2, %ymm0, %ymm0
100006340: c4 a1 7e 6f 54 59 ff        	vmovdqu	-1(%rcx,%r11,2), %ymm2
100006347: c5 e5 fe df                 	vpaddd	%ymm7, %ymm3, %ymm3
10000634b: c4 a1 7e 6f 64 59 1f        	vmovdqu	31(%rcx,%r11,2), %ymm4
100006352: c4 e2 5d 00 25 a5 0f 00 00  	vpshufb	4005(%rip), %ymm4, %ymm4
10000635b: c4 c2 6d 00 d3              	vpshufb	%ymm11, %ymm2, %ymm2
100006360: c4 e3 6d 02 d4 cc           	vpblendd	$204, %ymm4, %ymm2, %ymm2
100006366: c4 e3 fd 00 e2 d8           	vpermq	$216, %ymm2, %ymm4
10000636c: c4 c1 15 fe f2              	vpaddd	%ymm10, %ymm13, %ymm6
100006371: c4 e3 7d 39 e7 01           	vextracti128	$1, %ymm4, %xmm7
100006377: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
10000637c: c4 e3 fd 00 d2 db           	vpermq	$219, %ymm2, %ymm2
100006382: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
100006387: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
10000638c: c4 41 3d fe c1              	vpaddd	%ymm9, %ymm8, %ymm8
100006391: c4 e2 79 78 48 06           	vpbroadcastb	6(%rax), %xmm1
100006397: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
10000639c: c4 e2 75 40 e4              	vpmulld	%ymm4, %ymm1, %ymm4
1000063a1: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
1000063a5: c4 a1 7a 6f 64 59 0f        	vmovdqu	15(%rcx,%r11,2), %xmm4
1000063ac: c4 c2 59 00 e6              	vpshufb	%xmm14, %xmm4, %xmm4
1000063b1: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000063b6: c4 e2 75 40 d2              	vpmulld	%ymm2, %ymm1, %ymm2
1000063bb: c5 d5 fe d2                 	vpaddd	%ymm2, %ymm5, %ymm2
1000063bf: c4 62 75 40 ef              	vpmulld	%ymm7, %ymm1, %ymm13
1000063c4: c4 e2 75 40 cc              	vpmulld	%ymm4, %ymm1, %ymm1
1000063c9: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
1000063cd: c5 3d fe c3                 	vpaddd	%ymm3, %ymm8, %ymm8
1000063d1: c5 7d fe 94 24 e0 00 00 00  	vpaddd	224(%rsp), %ymm0, %ymm10
1000063da: c4 a1 7e 6f 0c 59           	vmovdqu	(%rcx,%r11,2), %ymm1
1000063e0: c4 a1 7e 6f 5c 59 20        	vmovdqu	32(%rcx,%r11,2), %ymm3
1000063e7: c4 e2 65 00 25 10 0f 00 00  	vpshufb	3856(%rip), %ymm3, %ymm4
1000063f0: c4 c2 75 00 eb              	vpshufb	%ymm11, %ymm1, %ymm5
1000063f5: c5 4d fe da                 	vpaddd	%ymm2, %ymm6, %ymm11
1000063f9: c4 e3 55 02 e4 cc           	vpblendd	$204, %ymm4, %ymm5, %ymm4
1000063ff: c4 e3 fd 00 ec d8           	vpermq	$216, %ymm4, %ymm5
100006405: c4 e2 65 00 1d 32 0f 00 00  	vpshufb	3890(%rip), %ymm3, %ymm3
10000640e: c4 e2 75 00 0d 49 0f 00 00  	vpshufb	3913(%rip), %ymm1, %ymm1
100006417: c4 e3 75 02 cb cc           	vpblendd	$204, %ymm3, %ymm1, %ymm1
10000641d: c5 fd 6f 84 24 00 01 00 00  	vmovdqa	256(%rsp), %ymm0
100006426: c5 7d fe a4 24 40 01 00 00  	vpaddd	320(%rsp), %ymm0, %ymm12
10000642f: c4 e3 fd 00 f1 d8           	vpermq	$216, %ymm1, %ymm6
100006435: c4 e2 7d 21 fd              	vpmovsxbd	%xmm5, %ymm7
10000643a: c4 e3 fd 00 e4 db           	vpermq	$219, %ymm4, %ymm4
100006440: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100006445: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
10000644b: c5 fd 6f 84 24 c0 00 00 00  	vmovdqa	192(%rsp), %ymm0
100006454: c5 7d fe 8c 24 20 01 00 00  	vpaddd	288(%rsp), %ymm0, %ymm9
10000645d: c4 a1 7a 6f 44 59 10        	vmovdqu	16(%rcx,%r11,2), %xmm0
100006464: c4 c2 79 00 d6              	vpshufb	%xmm14, %xmm0, %xmm2
100006469: c4 e2 79 78 58 07           	vpbroadcastb	7(%rax), %xmm3
10000646f: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
100006474: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006479: c4 e2 65 40 e4              	vpmulld	%ymm4, %ymm3, %ymm4
10000647e: c4 e2 65 40 ff              	vpmulld	%ymm7, %ymm3, %ymm7
100006483: c4 e2 65 40 ed              	vpmulld	%ymm5, %ymm3, %ymm5
100006488: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000648d: c4 e2 65 40 d2              	vpmulld	%ymm2, %ymm3, %ymm2
100006492: c4 e2 79 78 58 08           	vpbroadcastb	8(%rax), %xmm3
100006498: c5 05 fe b4 24 a0 00 00 00  	vpaddd	160(%rsp), %ymm15, %ymm14
1000064a1: c4 62 7d 21 fe              	vpmovsxbd	%xmm6, %ymm15
1000064a6: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000064ab: c4 42 65 40 ff              	vpmulld	%ymm15, %ymm3, %ymm15
1000064b0: c4 c1 45 fe ff              	vpaddd	%ymm15, %ymm7, %ymm7
1000064b5: c4 41 1d fe c9              	vpaddd	%ymm9, %ymm12, %ymm9
1000064ba: c4 e3 fd 00 c9 db           	vpermq	$219, %ymm1, %ymm1
1000064c0: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000064c5: c4 e2 65 40 c9              	vpmulld	%ymm1, %ymm3, %ymm1
1000064ca: c5 dd fe c9                 	vpaddd	%ymm1, %ymm4, %ymm1
1000064ce: c4 c1 0d fe e5              	vpaddd	%ymm13, %ymm14, %ymm4
1000064d3: c4 e3 7d 39 f6 01           	vextracti128	$1, %ymm6, %xmm6
1000064d9: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
1000064de: c4 e2 65 40 f6              	vpmulld	%ymm6, %ymm3, %ymm6
1000064e3: c5 d5 fe ee                 	vpaddd	%ymm6, %ymm5, %ymm5
1000064e7: c4 e2 79 00 05 00 0d 00 00  	vpshufb	3328(%rip), %xmm0, %xmm0
1000064f0: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
1000064f5: c4 e2 65 40 c0              	vpmulld	%ymm0, %ymm3, %ymm0
1000064fa: c5 ed fe c0                 	vpaddd	%ymm0, %ymm2, %ymm0
1000064fe: 48 8b 44 24 18              	movq	24(%rsp), %rax
100006503: c4 e2 79 78 10              	vpbroadcastb	(%rax), %xmm2
100006508: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
10000650d: c5 c5 fe da                 	vpaddd	%ymm2, %ymm7, %ymm3
100006511: c5 bd fe db                 	vpaddd	%ymm3, %ymm8, %ymm3
100006515: c5 f5 fe ca                 	vpaddd	%ymm2, %ymm1, %ymm1
100006519: c5 a5 fe c9                 	vpaddd	%ymm1, %ymm11, %ymm1
10000651d: c5 b5 fe e4                 	vpaddd	%ymm4, %ymm9, %ymm4
100006521: c5 d5 fe ea                 	vpaddd	%ymm2, %ymm5, %ymm5
100006525: c5 fd fe c2                 	vpaddd	%ymm2, %ymm0, %ymm0
100006529: c5 ad fe c0                 	vpaddd	%ymm0, %ymm10, %ymm0
10000652d: c5 fd 6f b4 24 60 02 00 00  	vmovdqa	608(%rsp), %ymm6
100006536: c4 e2 75 40 ce              	vpmulld	%ymm6, %ymm1, %ymm1
10000653b: c5 dd fe d5                 	vpaddd	%ymm5, %ymm4, %ymm2
10000653f: c4 e2 65 40 de              	vpmulld	%ymm6, %ymm3, %ymm3
100006544: c4 e2 7d 40 c6              	vpmulld	%ymm6, %ymm0, %ymm0
100006549: c4 e2 6d 40 d6              	vpmulld	%ymm6, %ymm2, %ymm2
10000654e: c5 dd 72 e3 1f              	vpsrad	$31, %ymm3, %ymm4
100006553: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
100006558: c5 e5 fe dc                 	vpaddd	%ymm4, %ymm3, %ymm3
10000655c: c5 dd 72 e1 1f              	vpsrad	$31, %ymm1, %ymm4
100006561: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
100006566: c5 e5 72 e3 0e              	vpsrad	$14, %ymm3, %ymm3
10000656b: c5 f5 fe cc                 	vpaddd	%ymm4, %ymm1, %ymm1
10000656f: c5 f5 72 e1 0e              	vpsrad	$14, %ymm1, %ymm1
100006574: c5 dd 72 e2 1f              	vpsrad	$31, %ymm2, %ymm4
100006579: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
10000657e: c5 ed fe d4                 	vpaddd	%ymm4, %ymm2, %ymm2
100006582: c5 ed 72 e2 0e              	vpsrad	$14, %ymm2, %ymm2
100006587: c5 dd 72 e0 1f              	vpsrad	$31, %ymm0, %ymm4
10000658c: c5 dd 72 d4 12              	vpsrld	$18, %ymm4, %ymm4
100006591: c5 fd fe c4                 	vpaddd	%ymm4, %ymm0, %ymm0
100006595: c5 fd 72 e0 0e              	vpsrad	$14, %ymm0, %ymm0
10000659a: c4 e2 7d 58 25 3d 27 00 00  	vpbroadcastd	10045(%rip), %ymm4
1000065a3: c4 e2 6d 39 d4              	vpminsd	%ymm4, %ymm2, %ymm2
1000065a8: c4 e2 75 39 cc              	vpminsd	%ymm4, %ymm1, %ymm1
1000065ad: c4 e2 65 39 dc              	vpminsd	%ymm4, %ymm3, %ymm3
1000065b2: c4 e2 7d 39 e4              	vpminsd	%ymm4, %ymm0, %ymm4
1000065b7: c4 e2 7d 58 2d 24 27 00 00  	vpbroadcastd	10020(%rip), %ymm5
1000065c0: c4 e2 75 3d c5              	vpmaxsd	%ymm5, %ymm1, %ymm0
1000065c5: c4 e2 6d 3d cd              	vpmaxsd	%ymm5, %ymm2, %ymm1
1000065ca: c5 f5 6b c0                 	vpackssdw	%ymm0, %ymm1, %ymm0
1000065ce: c4 e2 65 3d cd              	vpmaxsd	%ymm5, %ymm3, %ymm1
1000065d3: c4 e2 5d 3d d5              	vpmaxsd	%ymm5, %ymm4, %ymm2
1000065d8: c5 f5 6b ca                 	vpackssdw	%ymm2, %ymm1, %ymm1
1000065dc: c5 fd 6f b4 24 40 02 00 00  	vmovdqa	576(%rsp), %ymm6
1000065e5: c5 ed 73 d6 01              	vpsrlq	$1, %ymm6, %ymm2
1000065ea: c5 fd 6f ac 24 a0 02 00 00  	vmovdqa	672(%rsp), %ymm5
1000065f3: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
1000065f7: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000065fc: c5 fd 6f a4 24 80 02 00 00  	vmovdqa	640(%rsp), %ymm4
100006605: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006609: c4 c1 f9 7e d2              	vmovq	%xmm2, %r10
10000660e: c4 e3 f9 16 d0 01           	vpextrq	$1, %xmm2, %rax
100006614: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000661a: c4 c1 f9 7e d0              	vmovq	%xmm2, %r8
10000661f: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
100006625: c5 fd 6f bc 24 20 02 00 00  	vmovdqa	544(%rsp), %ymm7
10000662e: c5 ed 73 d7 01              	vpsrlq	$1, %ymm7, %ymm2
100006633: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006637: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000663c: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006640: c4 c1 f9 7e d5              	vmovq	%xmm2, %r13
100006645: c4 c3 f9 16 d6 01           	vpextrq	$1, %xmm2, %r14
10000664b: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006651: c4 e1 f9 7e d6              	vmovq	%xmm2, %rsi
100006656: c4 e3 f9 16 d7 01           	vpextrq	$1, %xmm2, %rdi
10000665c: c5 7d 6f 84 24 00 02 00 00  	vmovdqa	512(%rsp), %ymm8
100006665: c4 c1 6d 73 d0 01           	vpsrlq	$1, %ymm8, %ymm2
10000666b: c4 e3 fd 00 c0 d8           	vpermq	$216, %ymm0, %ymm0
100006671: c4 e3 fd 00 c9 d8           	vpermq	$216, %ymm1, %ymm1
100006677: c5 f5 63 c0                 	vpacksswb	%ymm0, %ymm1, %ymm0
10000667b: c5 7d 6f 8c 24 e0 01 00 00  	vmovdqa	480(%rsp), %ymm9
100006684: c4 c1 65 73 d1 01           	vpsrlq	$1, %ymm9, %ymm3
10000668a: c5 ed d4 cd                 	vpaddq	%ymm5, %ymm2, %ymm1
10000668e: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
100006693: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100006697: c4 e3 f9 16 8c 24 40 01 00 00 01    	vpextrq	$1, %xmm1, 320(%rsp)
1000066a2: c4 e1 f9 7e ca              	vmovq	%xmm1, %rdx
1000066a7: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
1000066ad: c5 f9 d6 8c 24 20 01 00 00  	vmovq	%xmm1, 288(%rsp)
1000066b6: c4 c3 f9 16 cc 01           	vpextrq	$1, %xmm1, %r12
1000066bc: c5 7d 6f 94 24 c0 01 00 00  	vmovdqa	448(%rsp), %ymm10
1000066c5: c4 c1 75 73 d2 01           	vpsrlq	$1, %ymm10, %ymm1
1000066cb: c5 e5 d4 d5                 	vpaddq	%ymm5, %ymm3, %ymm2
1000066cf: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000066d4: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000066d8: c4 83 79 14 04 17 00        	vpextrb	$0, %xmm0, (%r15,%r10)
1000066df: c4 e1 f9 7e d1              	vmovq	%xmm2, %rcx
1000066e4: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
1000066ea: c4 c3 79 14 04 07 01        	vpextrb	$1, %xmm0, (%r15,%rax)
1000066f1: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000066f7: c4 e3 f9 16 94 24 00 01 00 00 01    	vpextrq	$1, %xmm2, 256(%rsp)
100006702: c4 83 79 14 04 07 02        	vpextrb	$2, %xmm0, (%r15,%r8)
100006709: c4 e1 f9 7e d3              	vmovq	%xmm2, %rbx
10000670e: c5 fd 6f 9c 24 a0 01 00 00  	vmovdqa	416(%rsp), %ymm3
100006717: c5 ed 73 d3 01              	vpsrlq	$1, %ymm3, %ymm2
10000671c: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006720: c5 f5 d4 cd                 	vpaddq	%ymm5, %ymm1, %ymm1
100006724: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
100006729: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000672e: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006732: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100006736: c4 83 79 14 04 0f 03        	vpextrb	$3, %xmm0, (%r15,%r9)
10000673d: c4 c1 f9 7e c8              	vmovq	%xmm1, %r8
100006742: c4 83 79 14 04 2f 04        	vpextrb	$4, %xmm0, (%r15,%r13)
100006749: c4 e3 f9 16 8c 24 c0 00 00 00 01    	vpextrq	$1, %xmm1, 192(%rsp)
100006754: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
10000675a: c4 83 79 14 04 37 05        	vpextrb	$5, %xmm0, (%r15,%r14)
100006761: c4 c3 79 14 04 37 06        	vpextrb	$6, %xmm0, (%r15,%rsi)
100006768: c4 c1 f9 7e ce              	vmovq	%xmm1, %r14
10000676d: c4 e3 f9 16 8c 24 e0 00 00 00 01    	vpextrq	$1, %xmm1, 224(%rsp)
100006778: c4 c3 79 14 04 3f 07        	vpextrb	$7, %xmm0, (%r15,%rdi)
10000677f: c4 e3 f9 16 94 24 a0 00 00 00 01    	vpextrq	$1, %xmm2, 160(%rsp)
10000678a: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100006790: c4 c3 79 14 0c 17 00        	vpextrb	$0, %xmm1, (%r15,%rdx)
100006797: c4 e1 f9 7e d7              	vmovq	%xmm2, %rdi
10000679c: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000067a2: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
1000067aa: c4 c3 79 14 0c 07 01        	vpextrb	$1, %xmm1, (%r15,%rax)
1000067b1: c4 c1 f9 7e d5              	vmovq	%xmm2, %r13
1000067b6: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
1000067be: c4 c3 79 14 0c 07 02        	vpextrb	$2, %xmm1, (%r15,%rax)
1000067c5: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
1000067cb: c5 7d 6f 9c 24 80 01 00 00  	vmovdqa	384(%rsp), %ymm11
1000067d4: c4 c1 6d 73 d3 01           	vpsrlq	$1, %ymm11, %ymm2
1000067da: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
1000067de: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
1000067e3: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000067e7: c4 83 79 14 0c 27 03        	vpextrb	$3, %xmm1, (%r15,%r12)
1000067ee: c4 e1 f9 7e d2              	vmovq	%xmm2, %rdx
1000067f3: c4 c3 79 14 0c 0f 04        	vpextrb	$4, %xmm1, (%r15,%rcx)
1000067fa: c4 e3 f9 16 d1 01           	vpextrq	$1, %xmm2, %rcx
100006800: c4 83 79 14 0c 17 05        	vpextrb	$5, %xmm1, (%r15,%r10)
100006807: c4 c3 79 14 0c 1f 06        	vpextrb	$6, %xmm1, (%r15,%rbx)
10000680e: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006814: c4 e1 f9 7e d3              	vmovq	%xmm2, %rbx
100006819: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
10000681f: c5 7d 6f a4 24 60 01 00 00  	vmovdqa	352(%rsp), %ymm12
100006828: c4 c1 6d 73 d4 01           	vpsrlq	$1, %ymm12, %ymm2
10000682e: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006832: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006837: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
10000683b: 48 8b 84 24 00 01 00 00     	movq	256(%rsp), %rax
100006843: c4 c3 79 14 0c 07 07        	vpextrb	$7, %xmm1, (%r15,%rax)
10000684a: c4 e1 f9 7e d0              	vmovq	%xmm2, %rax
10000684f: c4 83 79 14 04 07 08        	vpextrb	$8, %xmm0, (%r15,%r8)
100006856: c4 c3 f9 16 d0 01           	vpextrq	$1, %xmm2, %r8
10000685c: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006862: 48 8b b4 24 c0 00 00 00     	movq	192(%rsp), %rsi
10000686a: c4 c3 79 14 04 37 09        	vpextrb	$9, %xmm0, (%r15,%rsi)
100006871: c4 83 79 14 04 37 0a        	vpextrb	$10, %xmm0, (%r15,%r14)
100006878: c4 c1 f9 7e d6              	vmovq	%xmm2, %r14
10000687d: c4 c3 f9 16 d4 01           	vpextrq	$1, %xmm2, %r12
100006883: c5 fd 6f 15 75 0a 00 00     	vmovdqa	2677(%rip), %ymm2
10000688b: 48 8b b4 24 e0 00 00 00     	movq	224(%rsp), %rsi
100006893: c4 c3 79 14 04 37 0b        	vpextrb	$11, %xmm0, (%r15,%rsi)
10000689a: c4 c3 79 14 04 3f 0c        	vpextrb	$12, %xmm0, (%r15,%rdi)
1000068a1: 48 8b b4 24 a0 00 00 00     	movq	160(%rsp), %rsi
1000068a9: c4 c3 79 14 04 37 0d        	vpextrb	$13, %xmm0, (%r15,%rsi)
1000068b0: c4 83 79 14 04 2f 0e        	vpextrb	$14, %xmm0, (%r15,%r13)
1000068b7: c4 83 79 14 04 0f 0f        	vpextrb	$15, %xmm0, (%r15,%r9)
1000068be: c4 c3 79 14 0c 17 08        	vpextrb	$8, %xmm1, (%r15,%rdx)
1000068c5: c4 c3 79 14 0c 0f 09        	vpextrb	$9, %xmm1, (%r15,%rcx)
1000068cc: c4 c3 79 14 0c 1f 0a        	vpextrb	$10, %xmm1, (%r15,%rbx)
1000068d3: c4 83 79 14 0c 17 0b        	vpextrb	$11, %xmm1, (%r15,%r10)
1000068da: c4 c3 79 14 0c 07 0c        	vpextrb	$12, %xmm1, (%r15,%rax)
1000068e1: c4 83 79 14 0c 07 0d        	vpextrb	$13, %xmm1, (%r15,%r8)
1000068e8: c4 83 79 14 0c 37 0e        	vpextrb	$14, %xmm1, (%r15,%r14)
1000068ef: c4 83 79 14 0c 27 0f        	vpextrb	$15, %xmm1, (%r15,%r12)
1000068f6: c4 e2 7d 59 05 e9 23 00 00  	vpbroadcastq	9193(%rip), %ymm0
1000068ff: c5 cd d4 f0                 	vpaddq	%ymm0, %ymm6, %ymm6
100006903: c5 fd 7f b4 24 40 02 00 00  	vmovdqa	%ymm6, 576(%rsp)
10000690c: c5 c5 d4 f8                 	vpaddq	%ymm0, %ymm7, %ymm7
100006910: c5 fd 7f bc 24 20 02 00 00  	vmovdqa	%ymm7, 544(%rsp)
100006919: c5 3d d4 c0                 	vpaddq	%ymm0, %ymm8, %ymm8
10000691d: c5 7d 7f 84 24 00 02 00 00  	vmovdqa	%ymm8, 512(%rsp)
100006926: c5 35 d4 c8                 	vpaddq	%ymm0, %ymm9, %ymm9
10000692a: c5 7d 7f 8c 24 e0 01 00 00  	vmovdqa	%ymm9, 480(%rsp)
100006933: c5 2d d4 d0                 	vpaddq	%ymm0, %ymm10, %ymm10
100006937: c5 7d 7f 94 24 c0 01 00 00  	vmovdqa	%ymm10, 448(%rsp)
100006940: c5 e5 d4 d8                 	vpaddq	%ymm0, %ymm3, %ymm3
100006944: c5 fd 7f 9c 24 a0 01 00 00  	vmovdqa	%ymm3, 416(%rsp)
10000694d: c5 25 d4 d8                 	vpaddq	%ymm0, %ymm11, %ymm11
100006951: c5 7d 7f 9c 24 80 01 00 00  	vmovdqa	%ymm11, 384(%rsp)
10000695a: c5 1d d4 e0                 	vpaddq	%ymm0, %ymm12, %ymm12
10000695e: c5 7d 7f a4 24 60 01 00 00  	vmovdqa	%ymm12, 352(%rsp)
100006967: 49 83 c3 20                 	addq	$32, %r11
10000696b: 49 81 fb e0 00 00 00        	cmpq	$224, %r11
100006972: 0f 85 68 f6 ff ff           	jne	-2456 <__ZN11LineNetwork7forwardEv+0x880>
100006978: ba c0 01 00 00              	movl	$448, %edx
10000697d: 44 8b 44 24 14              	movl	20(%rsp), %r8d
100006982: 48 8b 74 24 58              	movq	88(%rsp), %rsi
100006987: 4c 8b 6c 24 68              	movq	104(%rsp), %r13
10000698c: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100006991: eb 0f                       	jmp	15 <__ZN11LineNetwork7forwardEv+0x1242>
100006993: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000699d: 0f 1f 00                    	nopl	(%rax)
1000069a0: 31 d2                       	xorl	%edx, %edx
1000069a2: 48 83 44 24 08 02           	addq	$2, 8(%rsp)
1000069a8: 48 89 d0                    	movq	%rdx, %rax
1000069ab: 48 d1 e8                    	shrq	%rax
1000069ae: 4c 8b b4 24 98 00 00 00     	movq	152(%rsp), %r14
1000069b6: 4c 01 f0                    	addq	%r14, %rax
1000069b9: 4c 8d 0c c7                 	leaq	(%rdi,%rax,8), %r9
1000069bd: 4c 8b 54 24 20              	movq	32(%rsp), %r10
1000069c2: 4c 8b 5c 24 18              	movq	24(%rsp), %r11
1000069c7: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0x1288>
1000069c9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
1000069d0: 41 88 09                    	movb	%cl, (%r9)
1000069d3: 48 83 c2 02                 	addq	$2, %rdx
1000069d7: 49 83 c1 08                 	addq	$8, %r9
1000069db: 48 81 fa fd 01 00 00        	cmpq	$509, %rdx
1000069e2: 0f 83 78 f4 ff ff           	jae	-2952 <__ZN11LineNetwork7forwardEv+0x700>
1000069e8: 4c 8b 64 24 30              	movq	48(%rsp), %r12
1000069ed: 41 0f be 8c 14 fe fb ff ff  	movsbl	-1026(%r12,%rdx), %ecx
1000069f6: 41 0f be 02                 	movsbl	(%r10), %eax
1000069fa: 0f af c1                    	imull	%ecx, %eax
1000069fd: 41 0f be 8c 14 ff fb ff ff  	movsbl	-1025(%r12,%rdx), %ecx
100006a06: 41 0f be 5a 01              	movsbl	1(%r10), %ebx
100006a0b: 0f af d9                    	imull	%ecx, %ebx
100006a0e: 01 c3                       	addl	%eax, %ebx
100006a10: 41 0f be 8c 14 00 fc ff ff  	movsbl	-1024(%r12,%rdx), %ecx
100006a19: 41 0f be 42 02              	movsbl	2(%r10), %eax
100006a1e: 0f af c1                    	imull	%ecx, %eax
100006a21: 01 d8                       	addl	%ebx, %eax
100006a23: 41 0f be 8c 14 fe fd ff ff  	movsbl	-514(%r12,%rdx), %ecx
100006a2c: 41 0f be 5a 03              	movsbl	3(%r10), %ebx
100006a31: 0f af d9                    	imull	%ecx, %ebx
100006a34: 01 c3                       	addl	%eax, %ebx
100006a36: 41 0f be 8c 14 ff fd ff ff  	movsbl	-513(%r12,%rdx), %ecx
100006a3f: 41 0f be 42 04              	movsbl	4(%r10), %eax
100006a44: 0f af c1                    	imull	%ecx, %eax
100006a47: 01 d8                       	addl	%ebx, %eax
100006a49: 41 0f be 8c 14 00 fe ff ff  	movsbl	-512(%r12,%rdx), %ecx
100006a52: 41 0f be 5a 05              	movsbl	5(%r10), %ebx
100006a57: 0f af d9                    	imull	%ecx, %ebx
100006a5a: 01 c3                       	addl	%eax, %ebx
100006a5c: 41 0f be 4c 14 fe           	movsbl	-2(%r12,%rdx), %ecx
100006a62: 41 0f be 42 06              	movsbl	6(%r10), %eax
100006a67: 0f af c1                    	imull	%ecx, %eax
100006a6a: 01 d8                       	addl	%ebx, %eax
100006a6c: 41 0f be 4c 14 ff           	movsbl	-1(%r12,%rdx), %ecx
100006a72: 41 0f be 5a 07              	movsbl	7(%r10), %ebx
100006a77: 0f af d9                    	imull	%ecx, %ebx
100006a7a: 01 c3                       	addl	%eax, %ebx
100006a7c: 41 0f be 0c 14              	movsbl	(%r12,%rdx), %ecx
100006a81: 41 0f be 42 08              	movsbl	8(%r10), %eax
100006a86: 0f af c1                    	imull	%ecx, %eax
100006a89: 01 d8                       	addl	%ebx, %eax
100006a8b: 41 0f be 1b                 	movsbl	(%r11), %ebx
100006a8f: 01 c3                       	addl	%eax, %ebx
100006a91: 41 0f af d8                 	imull	%r8d, %ebx
100006a95: 89 d9                       	movl	%ebx, %ecx
100006a97: c1 f9 1f                    	sarl	$31, %ecx
100006a9a: c1 e9 12                    	shrl	$18, %ecx
100006a9d: 01 d9                       	addl	%ebx, %ecx
100006a9f: c1 f9 0e                    	sarl	$14, %ecx
100006aa2: 81 f9 80 00 00 00           	cmpl	$128, %ecx
100006aa8: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x134f>
100006aaa: b9 7f 00 00 00              	movl	$127, %ecx
100006aaf: 83 f9 81                    	cmpl	$-127, %ecx
100006ab2: 0f 8f 18 ff ff ff           	jg	-232 <__ZN11LineNetwork7forwardEv+0x1270>
100006ab8: b9 81 00 00 00              	movl	$129, %ecx
100006abd: e9 0e ff ff ff              	jmp	-242 <__ZN11LineNetwork7forwardEv+0x1270>
100006ac2: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100006ac6: 5b                          	popq	%rbx
100006ac7: 41 5c                       	popq	%r12
100006ac9: 41 5d                       	popq	%r13
100006acb: 41 5e                       	popq	%r14
100006acd: 41 5f                       	popq	%r15
100006acf: 5d                          	popq	%rbp
100006ad0: c5 f8 77                    	vzeroupper
100006ad3: c3                          	retq
100006ad4: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100006ade: 66 90                       	nop
100006ae0: 55                          	pushq	%rbp
100006ae1: 48 89 e5                    	movq	%rsp, %rbp
100006ae4: 5d                          	popq	%rbp
100006ae5: e9 d6 de ff ff              	jmp	-8490 <__ZN14ModelInterfaceD2Ev>
100006aea: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100006af0: 55                          	pushq	%rbp
100006af1: 48 89 e5                    	movq	%rsp, %rbp
100006af4: 53                          	pushq	%rbx
100006af5: 50                          	pushq	%rax
100006af6: 48 89 fb                    	movq	%rdi, %rbx
100006af9: e8 c2 de ff ff              	callq	-8510 <__ZN14ModelInterfaceD2Ev>
100006afe: 48 89 df                    	movq	%rbx, %rdi
100006b01: 48 83 c4 08                 	addq	$8, %rsp
100006b05: 5b                          	popq	%rbx
100006b06: 5d                          	popq	%rbp
100006b07: e9 aa 03 00 00              	jmp	938 <dyld_stub_binder+0x100006eb6>
100006b0c: 0f 1f 40 00                 	nopl	(%rax)
100006b10: 55                          	pushq	%rbp
100006b11: 48 89 e5                    	movq	%rsp, %rbp
100006b14: 41 57                       	pushq	%r15
100006b16: 41 56                       	pushq	%r14
100006b18: 41 54                       	pushq	%r12
100006b1a: 53                          	pushq	%rbx
100006b1b: 0f be 07                    	movsbl	(%rdi), %eax
100006b1e: 0f be 1e                    	movsbl	(%rsi), %ebx
100006b21: 0f af d8                    	imull	%eax, %ebx
100006b24: 0f be 47 01                 	movsbl	1(%rdi), %eax
100006b28: 0f be 56 01                 	movsbl	1(%rsi), %edx
100006b2c: 0f af d0                    	imull	%eax, %edx
100006b2f: 0f be 4f 02                 	movsbl	2(%rdi), %ecx
100006b33: 44 0f be 7e 02              	movsbl	2(%rsi), %r15d
100006b38: 44 0f af f9                 	imull	%ecx, %r15d
100006b3c: 0f be 4f 03                 	movsbl	3(%rdi), %ecx
100006b40: 44 0f be 5e 03              	movsbl	3(%rsi), %r11d
100006b45: 44 0f af d9                 	imull	%ecx, %r11d
100006b49: 0f be 4f 04                 	movsbl	4(%rdi), %ecx
100006b4d: 44 0f be 56 04              	movsbl	4(%rsi), %r10d
100006b52: 44 0f af d1                 	imull	%ecx, %r10d
100006b56: 0f be 4f 05                 	movsbl	5(%rdi), %ecx
100006b5a: 44 0f be 4e 05              	movsbl	5(%rsi), %r9d
100006b5f: 44 0f af c9                 	imull	%ecx, %r9d
100006b63: 0f be 4f 06                 	movsbl	6(%rdi), %ecx
100006b67: 44 0f be 46 06              	movsbl	6(%rsi), %r8d
100006b6c: 44 0f af c1                 	imull	%ecx, %r8d
100006b70: 0f be 4f 07                 	movsbl	7(%rdi), %ecx
100006b74: 44 0f be 76 07              	movsbl	7(%rsi), %r14d
100006b79: 44 0f af f1                 	imull	%ecx, %r14d
100006b7d: 0f be 47 10                 	movsbl	16(%rdi), %eax
100006b81: 0f be 4e 10                 	movsbl	16(%rsi), %ecx
100006b85: 0f af c8                    	imull	%eax, %ecx
100006b88: 01 d9                       	addl	%ebx, %ecx
100006b8a: 0f be 47 11                 	movsbl	17(%rdi), %eax
100006b8e: 0f be 5e 11                 	movsbl	17(%rsi), %ebx
100006b92: 0f af d8                    	imull	%eax, %ebx
100006b95: 01 d3                       	addl	%edx, %ebx
100006b97: 0f be 47 12                 	movsbl	18(%rdi), %eax
100006b9b: 0f be 56 12                 	movsbl	18(%rsi), %edx
100006b9f: 0f af d0                    	imull	%eax, %edx
100006ba2: 44 01 fa                    	addl	%r15d, %edx
100006ba5: 44 0f be 7f 13              	movsbl	19(%rdi), %r15d
100006baa: 44 0f be 66 13              	movsbl	19(%rsi), %r12d
100006baf: 45 0f af e7                 	imull	%r15d, %r12d
100006bb3: 45 01 dc                    	addl	%r11d, %r12d
100006bb6: 44 0f be 7f 14              	movsbl	20(%rdi), %r15d
100006bbb: 44 0f be 5e 14              	movsbl	20(%rsi), %r11d
100006bc0: 45 0f af df                 	imull	%r15d, %r11d
100006bc4: 45 01 d3                    	addl	%r10d, %r11d
100006bc7: 44 0f be 7f 15              	movsbl	21(%rdi), %r15d
100006bcc: 44 0f be 56 15              	movsbl	21(%rsi), %r10d
100006bd1: 45 0f af d7                 	imull	%r15d, %r10d
100006bd5: 45 01 ca                    	addl	%r9d, %r10d
100006bd8: 44 0f be 7f 16              	movsbl	22(%rdi), %r15d
100006bdd: 44 0f be 4e 16              	movsbl	22(%rsi), %r9d
100006be2: 45 0f af cf                 	imull	%r15d, %r9d
100006be6: 45 01 c1                    	addl	%r8d, %r9d
100006be9: 44 0f be 7f 17              	movsbl	23(%rdi), %r15d
100006bee: 44 0f be 46 17              	movsbl	23(%rsi), %r8d
100006bf3: 45 0f af c7                 	imull	%r15d, %r8d
100006bf7: 45 01 f0                    	addl	%r14d, %r8d
100006bfa: 44 0f be 77 20              	movsbl	32(%rdi), %r14d
100006bff: 0f be 46 20                 	movsbl	32(%rsi), %eax
100006c03: 41 0f af c6                 	imull	%r14d, %eax
100006c07: 01 c8                       	addl	%ecx, %eax
100006c09: 44 0f be 77 21              	movsbl	33(%rdi), %r14d
100006c0e: 0f be 4e 21                 	movsbl	33(%rsi), %ecx
100006c12: 41 0f af ce                 	imull	%r14d, %ecx
100006c16: 01 d9                       	addl	%ebx, %ecx
100006c18: 01 c1                       	addl	%eax, %ecx
100006c1a: 0f be 47 22                 	movsbl	34(%rdi), %eax
100006c1e: 0f be 5e 22                 	movsbl	34(%rsi), %ebx
100006c22: 0f af d8                    	imull	%eax, %ebx
100006c25: 01 d3                       	addl	%edx, %ebx
100006c27: 01 cb                       	addl	%ecx, %ebx
100006c29: 0f be 47 23                 	movsbl	35(%rdi), %eax
100006c2d: 0f be 4e 23                 	movsbl	35(%rsi), %ecx
100006c31: 0f af c8                    	imull	%eax, %ecx
100006c34: 44 01 e1                    	addl	%r12d, %ecx
100006c37: 01 d9                       	addl	%ebx, %ecx
100006c39: 0f be 47 24                 	movsbl	36(%rdi), %eax
100006c3d: 0f be 56 24                 	movsbl	36(%rsi), %edx
100006c41: 0f af d0                    	imull	%eax, %edx
100006c44: 44 01 da                    	addl	%r11d, %edx
100006c47: 01 ca                       	addl	%ecx, %edx
100006c49: 0f be 47 25                 	movsbl	37(%rdi), %eax
100006c4d: 0f be 4e 25                 	movsbl	37(%rsi), %ecx
100006c51: 0f af c8                    	imull	%eax, %ecx
100006c54: 44 01 d1                    	addl	%r10d, %ecx
100006c57: 01 d1                       	addl	%edx, %ecx
100006c59: 0f be 47 26                 	movsbl	38(%rdi), %eax
100006c5d: 0f be 56 26                 	movsbl	38(%rsi), %edx
100006c61: 0f af d0                    	imull	%eax, %edx
100006c64: 44 01 ca                    	addl	%r9d, %edx
100006c67: 01 ca                       	addl	%ecx, %edx
100006c69: 0f be 4f 27                 	movsbl	39(%rdi), %ecx
100006c6d: 0f be 46 27                 	movsbl	39(%rsi), %eax
100006c71: 0f af c1                    	imull	%ecx, %eax
100006c74: 44 01 c0                    	addl	%r8d, %eax
100006c77: 01 d0                       	addl	%edx, %eax
100006c79: 5b                          	popq	%rbx
100006c7a: 41 5c                       	popq	%r12
100006c7c: 41 5e                       	popq	%r14
100006c7e: 41 5f                       	popq	%r15
100006c80: 5d                          	popq	%rbp
100006c81: c3                          	retq
100006c82: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100006c8c: 0f 1f 40 00                 	nopl	(%rax)
100006c90: 55                          	pushq	%rbp
100006c91: 48 89 e5                    	movq	%rsp, %rbp
100006c94: 41 57                       	pushq	%r15
100006c96: 41 56                       	pushq	%r14
100006c98: 41 54                       	pushq	%r12
100006c9a: 53                          	pushq	%rbx
100006c9b: 0f be 07                    	movsbl	(%rdi), %eax
100006c9e: 0f be 1e                    	movsbl	(%rsi), %ebx
100006ca1: 0f af d8                    	imull	%eax, %ebx
100006ca4: 0f be 47 01                 	movsbl	1(%rdi), %eax
100006ca8: 0f be 56 01                 	movsbl	1(%rsi), %edx
100006cac: 0f af d0                    	imull	%eax, %edx
100006caf: 0f be 4f 02                 	movsbl	2(%rdi), %ecx
100006cb3: 44 0f be 7e 02              	movsbl	2(%rsi), %r15d
100006cb8: 44 0f af f9                 	imull	%ecx, %r15d
100006cbc: 0f be 4f 03                 	movsbl	3(%rdi), %ecx
100006cc0: 44 0f be 5e 03              	movsbl	3(%rsi), %r11d
100006cc5: 44 0f af d9                 	imull	%ecx, %r11d
100006cc9: 0f be 4f 04                 	movsbl	4(%rdi), %ecx
100006ccd: 44 0f be 56 04              	movsbl	4(%rsi), %r10d
100006cd2: 44 0f af d1                 	imull	%ecx, %r10d
100006cd6: 0f be 4f 05                 	movsbl	5(%rdi), %ecx
100006cda: 44 0f be 4e 05              	movsbl	5(%rsi), %r9d
100006cdf: 44 0f af c9                 	imull	%ecx, %r9d
100006ce3: 0f be 4f 06                 	movsbl	6(%rdi), %ecx
100006ce7: 44 0f be 46 06              	movsbl	6(%rsi), %r8d
100006cec: 44 0f af c1                 	imull	%ecx, %r8d
100006cf0: 0f be 4f 07                 	movsbl	7(%rdi), %ecx
100006cf4: 44 0f be 76 07              	movsbl	7(%rsi), %r14d
100006cf9: 44 0f af f1                 	imull	%ecx, %r14d
100006cfd: 0f be 47 20                 	movsbl	32(%rdi), %eax
100006d01: 0f be 4e 20                 	movsbl	32(%rsi), %ecx
100006d05: 0f af c8                    	imull	%eax, %ecx
100006d08: 01 d9                       	addl	%ebx, %ecx
100006d0a: 0f be 47 21                 	movsbl	33(%rdi), %eax
100006d0e: 0f be 5e 21                 	movsbl	33(%rsi), %ebx
100006d12: 0f af d8                    	imull	%eax, %ebx
100006d15: 01 d3                       	addl	%edx, %ebx
100006d17: 0f be 47 22                 	movsbl	34(%rdi), %eax
100006d1b: 0f be 56 22                 	movsbl	34(%rsi), %edx
100006d1f: 0f af d0                    	imull	%eax, %edx
100006d22: 44 01 fa                    	addl	%r15d, %edx
100006d25: 44 0f be 7f 23              	movsbl	35(%rdi), %r15d
100006d2a: 44 0f be 66 23              	movsbl	35(%rsi), %r12d
100006d2f: 45 0f af e7                 	imull	%r15d, %r12d
100006d33: 45 01 dc                    	addl	%r11d, %r12d
100006d36: 44 0f be 7f 24              	movsbl	36(%rdi), %r15d
100006d3b: 44 0f be 5e 24              	movsbl	36(%rsi), %r11d
100006d40: 45 0f af df                 	imull	%r15d, %r11d
100006d44: 45 01 d3                    	addl	%r10d, %r11d
100006d47: 44 0f be 7f 25              	movsbl	37(%rdi), %r15d
100006d4c: 44 0f be 56 25              	movsbl	37(%rsi), %r10d
100006d51: 45 0f af d7                 	imull	%r15d, %r10d
100006d55: 45 01 ca                    	addl	%r9d, %r10d
100006d58: 44 0f be 7f 26              	movsbl	38(%rdi), %r15d
100006d5d: 44 0f be 4e 26              	movsbl	38(%rsi), %r9d
100006d62: 45 0f af cf                 	imull	%r15d, %r9d
100006d66: 45 01 c1                    	addl	%r8d, %r9d
100006d69: 44 0f be 7f 27              	movsbl	39(%rdi), %r15d
100006d6e: 44 0f be 46 27              	movsbl	39(%rsi), %r8d
100006d73: 45 0f af c7                 	imull	%r15d, %r8d
100006d77: 45 01 f0                    	addl	%r14d, %r8d
100006d7a: 44 0f be 77 30              	movsbl	48(%rdi), %r14d
100006d7f: 0f be 46 30                 	movsbl	48(%rsi), %eax
100006d83: 41 0f af c6                 	imull	%r14d, %eax
100006d87: 01 c8                       	addl	%ecx, %eax
100006d89: 44 0f be 77 31              	movsbl	49(%rdi), %r14d
100006d8e: 0f be 4e 31                 	movsbl	49(%rsi), %ecx
100006d92: 41 0f af ce                 	imull	%r14d, %ecx
100006d96: 01 d9                       	addl	%ebx, %ecx
100006d98: 01 c1                       	addl	%eax, %ecx
100006d9a: 0f be 47 32                 	movsbl	50(%rdi), %eax
100006d9e: 0f be 5e 32                 	movsbl	50(%rsi), %ebx
100006da2: 0f af d8                    	imull	%eax, %ebx
100006da5: 01 d3                       	addl	%edx, %ebx
100006da7: 01 cb                       	addl	%ecx, %ebx
100006da9: 0f be 47 33                 	movsbl	51(%rdi), %eax
100006dad: 0f be 4e 33                 	movsbl	51(%rsi), %ecx
100006db1: 0f af c8                    	imull	%eax, %ecx
100006db4: 44 01 e1                    	addl	%r12d, %ecx
100006db7: 01 d9                       	addl	%ebx, %ecx
100006db9: 0f be 47 34                 	movsbl	52(%rdi), %eax
100006dbd: 0f be 56 34                 	movsbl	52(%rsi), %edx
100006dc1: 0f af d0                    	imull	%eax, %edx
100006dc4: 44 01 da                    	addl	%r11d, %edx
100006dc7: 01 ca                       	addl	%ecx, %edx
100006dc9: 0f be 47 35                 	movsbl	53(%rdi), %eax
100006dcd: 0f be 4e 35                 	movsbl	53(%rsi), %ecx
100006dd1: 0f af c8                    	imull	%eax, %ecx
100006dd4: 44 01 d1                    	addl	%r10d, %ecx
100006dd7: 01 d1                       	addl	%edx, %ecx
100006dd9: 0f be 47 36                 	movsbl	54(%rdi), %eax
100006ddd: 0f be 56 36                 	movsbl	54(%rsi), %edx
100006de1: 0f af d0                    	imull	%eax, %edx
100006de4: 44 01 ca                    	addl	%r9d, %edx
100006de7: 01 ca                       	addl	%ecx, %edx
100006de9: 0f be 4f 37                 	movsbl	55(%rdi), %ecx
100006ded: 0f be 46 37                 	movsbl	55(%rsi), %eax
100006df1: 0f af c1                    	imull	%ecx, %eax
100006df4: 44 01 c0                    	addl	%r8d, %eax
100006df7: 01 d0                       	addl	%edx, %eax
100006df9: 5b                          	popq	%rbx
100006dfa: 41 5c                       	popq	%r12
100006dfc: 41 5e                       	popq	%r14
100006dfe: 41 5f                       	popq	%r15
100006e00: 5d                          	popq	%rbp
100006e01: c3                          	retq

Disassembly of section __TEXT,__stubs:

0000000100006e02 __stubs:
100006e02: ff 25 f8 31 00 00           	jmpq	*12792(%rip)
100006e08: ff 25 fa 31 00 00           	jmpq	*12794(%rip)
100006e0e: ff 25 fc 31 00 00           	jmpq	*12796(%rip)
100006e14: ff 25 fe 31 00 00           	jmpq	*12798(%rip)
100006e1a: ff 25 00 32 00 00           	jmpq	*12800(%rip)
100006e20: ff 25 02 32 00 00           	jmpq	*12802(%rip)
100006e26: ff 25 04 32 00 00           	jmpq	*12804(%rip)
100006e2c: ff 25 06 32 00 00           	jmpq	*12806(%rip)
100006e32: ff 25 08 32 00 00           	jmpq	*12808(%rip)
100006e38: ff 25 0a 32 00 00           	jmpq	*12810(%rip)
100006e3e: ff 25 0c 32 00 00           	jmpq	*12812(%rip)
100006e44: ff 25 0e 32 00 00           	jmpq	*12814(%rip)
100006e4a: ff 25 10 32 00 00           	jmpq	*12816(%rip)
100006e50: ff 25 12 32 00 00           	jmpq	*12818(%rip)
100006e56: ff 25 14 32 00 00           	jmpq	*12820(%rip)
100006e5c: ff 25 16 32 00 00           	jmpq	*12822(%rip)
100006e62: ff 25 18 32 00 00           	jmpq	*12824(%rip)
100006e68: ff 25 1a 32 00 00           	jmpq	*12826(%rip)
100006e6e: ff 25 1c 32 00 00           	jmpq	*12828(%rip)
100006e74: ff 25 1e 32 00 00           	jmpq	*12830(%rip)
100006e7a: ff 25 20 32 00 00           	jmpq	*12832(%rip)
100006e80: ff 25 22 32 00 00           	jmpq	*12834(%rip)
100006e86: ff 25 24 32 00 00           	jmpq	*12836(%rip)
100006e8c: ff 25 26 32 00 00           	jmpq	*12838(%rip)
100006e92: ff 25 28 32 00 00           	jmpq	*12840(%rip)
100006e98: ff 25 2a 32 00 00           	jmpq	*12842(%rip)
100006e9e: ff 25 2c 32 00 00           	jmpq	*12844(%rip)
100006ea4: ff 25 2e 32 00 00           	jmpq	*12846(%rip)
100006eaa: ff 25 30 32 00 00           	jmpq	*12848(%rip)
100006eb0: ff 25 32 32 00 00           	jmpq	*12850(%rip)
100006eb6: ff 25 34 32 00 00           	jmpq	*12852(%rip)
100006ebc: ff 25 36 32 00 00           	jmpq	*12854(%rip)
100006ec2: ff 25 38 32 00 00           	jmpq	*12856(%rip)
100006ec8: ff 25 3a 32 00 00           	jmpq	*12858(%rip)
100006ece: ff 25 3c 32 00 00           	jmpq	*12860(%rip)
100006ed4: ff 25 3e 32 00 00           	jmpq	*12862(%rip)
100006eda: ff 25 40 32 00 00           	jmpq	*12864(%rip)
100006ee0: ff 25 42 32 00 00           	jmpq	*12866(%rip)

Disassembly of section __TEXT,__stub_helper:

0000000100006ee8 __stub_helper:
100006ee8: 4c 8d 1d 41 32 00 00        	leaq	12865(%rip), %r11
100006eef: 41 53                       	pushq	%r11
100006ef1: ff 25 71 21 00 00           	jmpq	*8561(%rip)
100006ef7: 90                          	nop
100006ef8: 68 4e 01 00 00              	pushq	$334
100006efd: e9 e6 ff ff ff              	jmp	-26 <__stub_helper>
100006f02: 68 9c 02 00 00              	pushq	$668
100006f07: e9 dc ff ff ff              	jmp	-36 <__stub_helper>
100006f0c: 68 17 00 00 00              	pushq	$23
100006f11: e9 d2 ff ff ff              	jmp	-46 <__stub_helper>
100006f16: 68 7a 00 00 00              	pushq	$122
100006f1b: e9 c8 ff ff ff              	jmp	-56 <__stub_helper>
100006f20: 68 9b 00 00 00              	pushq	$155
100006f25: e9 be ff ff ff              	jmp	-66 <__stub_helper>
100006f2a: 68 2e 03 00 00              	pushq	$814
100006f2f: e9 b4 ff ff ff              	jmp	-76 <__stub_helper>
100006f34: 68 b9 01 00 00              	pushq	$441
100006f39: e9 aa ff ff ff              	jmp	-86 <__stub_helper>
100006f3e: 68 07 02 00 00              	pushq	$519
100006f43: e9 a0 ff ff ff              	jmp	-96 <__stub_helper>
100006f48: 68 b4 02 00 00              	pushq	$692
100006f4d: e9 96 ff ff ff              	jmp	-106 <__stub_helper>
100006f52: 68 c4 00 00 00              	pushq	$196
100006f57: e9 8c ff ff ff              	jmp	-116 <__stub_helper>
100006f5c: 68 e5 00 00 00              	pushq	$229
100006f61: e9 82 ff ff ff              	jmp	-126 <__stub_helper>
100006f66: 68 05 01 00 00              	pushq	$261
100006f6b: e9 78 ff ff ff              	jmp	-136 <__stub_helper>
100006f70: 68 27 01 00 00              	pushq	$295
100006f75: e9 6e ff ff ff              	jmp	-146 <__stub_helper>
100006f7a: 68 f6 02 00 00              	pushq	$758
100006f7f: e9 64 ff ff ff              	jmp	-156 <__stub_helper>
100006f84: 68 11 03 00 00              	pushq	$785
100006f89: e9 5a ff ff ff              	jmp	-166 <__stub_helper>
100006f8e: 68 57 03 00 00              	pushq	$855
100006f93: e9 50 ff ff ff              	jmp	-176 <__stub_helper>
100006f98: 68 86 03 00 00              	pushq	$902
100006f9d: e9 46 ff ff ff              	jmp	-186 <__stub_helper>
100006fa2: 68 ac 03 00 00              	pushq	$940
100006fa7: e9 3c ff ff ff              	jmp	-196 <__stub_helper>
100006fac: 68 00 04 00 00              	pushq	$1024
100006fb1: e9 32 ff ff ff              	jmp	-206 <__stub_helper>
100006fb6: 68 55 04 00 00              	pushq	$1109
100006fbb: e9 28 ff ff ff              	jmp	-216 <__stub_helper>
100006fc0: 68 aa 04 00 00              	pushq	$1194
100006fc5: e9 1e ff ff ff              	jmp	-226 <__stub_helper>
100006fca: 68 f1 04 00 00              	pushq	$1265
100006fcf: e9 14 ff ff ff              	jmp	-236 <__stub_helper>
100006fd4: 68 35 05 00 00              	pushq	$1333
100006fd9: e9 0a ff ff ff              	jmp	-246 <__stub_helper>
100006fde: 68 63 05 00 00              	pushq	$1379
100006fe3: e9 00 ff ff ff              	jmp	-256 <__stub_helper>
100006fe8: 68 81 05 00 00              	pushq	$1409
100006fed: e9 f6 fe ff ff              	jmp	-266 <__stub_helper>
100006ff2: 68 c2 05 00 00              	pushq	$1474
100006ff7: e9 ec fe ff ff              	jmp	-276 <__stub_helper>
100006ffc: 68 e6 05 00 00              	pushq	$1510
100007001: e9 e2 fe ff ff              	jmp	-286 <__stub_helper>
100007006: 68 05 06 00 00              	pushq	$1541
10000700b: e9 d8 fe ff ff              	jmp	-296 <__stub_helper>
100007010: 68 24 06 00 00              	pushq	$1572
100007015: e9 ce fe ff ff              	jmp	-306 <__stub_helper>
10000701a: 68 3d 06 00 00              	pushq	$1597
10000701f: e9 c4 fe ff ff              	jmp	-316 <__stub_helper>
100007024: 68 58 06 00 00              	pushq	$1624
100007029: e9 ba fe ff ff              	jmp	-326 <__stub_helper>
10000702e: 68 00 00 00 00              	pushq	$0
100007033: e9 b0 fe ff ff              	jmp	-336 <__stub_helper>
100007038: 68 71 06 00 00              	pushq	$1649
10000703d: e9 a6 fe ff ff              	jmp	-346 <__stub_helper>
100007042: 68 8b 06 00 00              	pushq	$1675
100007047: e9 9c fe ff ff              	jmp	-356 <__stub_helper>
10000704c: 68 9b 06 00 00              	pushq	$1691
100007051: e9 92 fe ff ff              	jmp	-366 <__stub_helper>
