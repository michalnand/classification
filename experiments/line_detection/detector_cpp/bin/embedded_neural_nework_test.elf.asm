
bin/embedded_neural_nework_test.elf:	file format Mach-O 64-bit x86-64


Disassembly of section __TEXT,__text:

0000000100001b30 __Z8get_timev:
100001b30: 55                          	pushq	%rbp
100001b31: 48 89 e5                    	movq	%rsp, %rbp
100001b34: e8 6f 53 00 00              	callq	21359 <dyld_stub_binder+0x100006ea8>
100001b39: c4 e1 fb 2a c0              	vcvtsi2sd	%rax, %xmm0, %xmm0
100001b3e: c5 fb 5e 05 3a 55 00 00     	vdivsd	21818(%rip), %xmm0, %xmm0
100001b46: 5d                          	popq	%rbp
100001b47: c3                          	retq
100001b48: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)

0000000100001b50 __Z14get_predictionRN2cv3MatER11LineNetworkf:
100001b50: 55                          	pushq	%rbp
100001b51: 48 89 e5                    	movq	%rsp, %rbp
100001b54: 41 57                       	pushq	%r15
100001b56: 41 56                       	pushq	%r14
100001b58: 41 55                       	pushq	%r13
100001b5a: 41 54                       	pushq	%r12
100001b5c: 53                          	pushq	%rbx
100001b5d: 48 81 ec 18 01 00 00        	subq	$280, %rsp
100001b64: c5 fa 11 45 b0              	vmovss	%xmm0, -80(%rbp)
100001b69: 49 89 d5                    	movq	%rdx, %r13
100001b6c: 49 89 f4                    	movq	%rsi, %r12
100001b6f: 49 89 fe                    	movq	%rdi, %r14
100001b72: 48 8b 05 e7 74 00 00        	movq	29927(%rip), %rax
100001b79: 48 8b 00                    	movq	(%rax), %rax
100001b7c: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100001b80: 8b 46 08                    	movl	8(%rsi), %eax
100001b83: 8b 4e 0c                    	movl	12(%rsi), %ecx
100001b86: c7 85 e0 fe ff ff 00 00 ff 42       	movl	$1124007936, -288(%rbp)
100001b90: 48 8d 95 e8 fe ff ff        	leaq	-280(%rbp), %rdx
100001b97: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001b9b: c5 fc 11 85 e4 fe ff ff     	vmovups	%ymm0, -284(%rbp)
100001ba3: c5 fc 11 85 00 ff ff ff     	vmovups	%ymm0, -256(%rbp)
100001bab: 48 89 95 20 ff ff ff        	movq	%rdx, -224(%rbp)
100001bb2: 48 8d 95 30 ff ff ff        	leaq	-208(%rbp), %rdx
100001bb9: 48 89 95 28 ff ff ff        	movq	%rdx, -216(%rbp)
100001bc0: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001bc4: c5 f8 11 85 30 ff ff ff     	vmovups	%xmm0, -208(%rbp)
100001bcc: 89 4d b8                    	movl	%ecx, -72(%rbp)
100001bcf: 89 45 bc                    	movl	%eax, -68(%rbp)
100001bd2: 4c 8d bd e0 fe ff ff        	leaq	-288(%rbp), %r15
100001bd9: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
100001bdd: 4c 89 ff                    	movq	%r15, %rdi
100001be0: be 02 00 00 00              	movl	$2, %esi
100001be5: 31 c9                       	xorl	%ecx, %ecx
100001be7: c5 f8 77                    	vzeroupper
100001bea: e8 4d 52 00 00              	callq	21069 <dyld_stub_binder+0x100006e3c>
100001bef: 48 c7 85 50 ff ff ff 00 00 00 00    	movq	$0, -176(%rbp)
100001bfa: c7 85 40 ff ff ff 00 00 01 01       	movl	$16842752, -192(%rbp)
100001c04: 4c 89 a5 48 ff ff ff        	movq	%r12, -184(%rbp)
100001c0b: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
100001c13: c7 45 b8 00 00 01 02        	movl	$33619968, -72(%rbp)
100001c1a: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
100001c1e: 48 8d bd 40 ff ff ff        	leaq	-192(%rbp), %rdi
100001c25: 48 8d 75 b8                 	leaq	-72(%rbp), %rsi
100001c29: ba 06 00 00 00              	movl	$6, %edx
100001c2e: 31 c9                       	xorl	%ecx, %ecx
100001c30: e8 31 52 00 00              	callq	21041 <dyld_stub_binder+0x100006e66>
100001c35: 41 8b 44 24 08              	movl	8(%r12), %eax
100001c3a: 41 8b 4c 24 0c              	movl	12(%r12), %ecx
100001c3f: c7 85 40 ff ff ff 00 00 ff 42       	movl	$1124007936, -192(%rbp)
100001c49: 48 8d 95 48 ff ff ff        	leaq	-184(%rbp), %rdx
100001c50: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001c54: c5 fc 11 85 44 ff ff ff     	vmovups	%ymm0, -188(%rbp)
100001c5c: c5 fc 11 85 60 ff ff ff     	vmovups	%ymm0, -160(%rbp)
100001c64: 48 89 55 80                 	movq	%rdx, -128(%rbp)
100001c68: 48 8d 55 90                 	leaq	-112(%rbp), %rdx
100001c6c: 48 89 55 88                 	movq	%rdx, -120(%rbp)
100001c70: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001c74: c5 f8 11 45 90              	vmovups	%xmm0, -112(%rbp)
100001c79: 89 4d b8                    	movl	%ecx, -72(%rbp)
100001c7c: 89 45 bc                    	movl	%eax, -68(%rbp)
100001c7f: 4c 8d a5 40 ff ff ff        	leaq	-192(%rbp), %r12
100001c86: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
100001c8a: 4c 89 e7                    	movq	%r12, %rdi
100001c8d: be 02 00 00 00              	movl	$2, %esi
100001c92: 31 c9                       	xorl	%ecx, %ecx
100001c94: c5 f8 77                    	vzeroupper
100001c97: e8 a0 51 00 00              	callq	20896 <dyld_stub_binder+0x100006e3c>
100001c9c: 48 c7 45 c8 00 00 00 00     	movq	$0, -56(%rbp)
100001ca4: c7 45 b8 00 00 01 01        	movl	$16842752, -72(%rbp)
100001cab: 4c 89 7d c0                 	movq	%r15, -64(%rbp)
100001caf: 48 c7 85 d8 fe ff ff 00 00 00 00    	movq	$0, -296(%rbp)
100001cba: c7 85 c8 fe ff ff 00 00 01 02       	movl	$33619968, -312(%rbp)
100001cc4: 4c 89 a5 d0 fe ff ff        	movq	%r12, -304(%rbp)
100001ccb: 41 8b 45 0c                 	movl	12(%r13), %eax
100001ccf: 41 8b 4d 10                 	movl	16(%r13), %ecx
100001cd3: 89 4d a0                    	movl	%ecx, -96(%rbp)
100001cd6: 89 45 a4                    	movl	%eax, -92(%rbp)
100001cd9: 48 8d 7d b8                 	leaq	-72(%rbp), %rdi
100001cdd: 48 8d b5 c8 fe ff ff        	leaq	-312(%rbp), %rsi
100001ce4: 48 8d 55 a0                 	leaq	-96(%rbp), %rdx
100001ce8: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001cec: c5 f0 57 c9                 	vxorps	%xmm1, %xmm1, %xmm1
100001cf0: b9 03 00 00 00              	movl	$3, %ecx
100001cf5: e8 5a 51 00 00              	callq	20826 <dyld_stub_binder+0x100006e54>
100001cfa: 41 8b 55 0c                 	movl	12(%r13), %edx
100001cfe: 85 d2                       	testl	%edx, %edx
100001d00: 74 7d                       	je	125 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x22f>
100001d02: 41 8b 75 10                 	movl	16(%r13), %esi
100001d06: 45 31 c0                    	xorl	%r8d, %r8d
100001d09: 45 31 c9                    	xorl	%r9d, %r9d
100001d0c: 85 f6                       	testl	%esi, %esi
100001d0e: 75 0e                       	jne	14 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x1ce>
100001d10: 31 f6                       	xorl	%esi, %esi
100001d12: 41 ff c0                    	incl	%r8d
100001d15: 41 39 d0                    	cmpl	%edx, %r8d
100001d18: 73 65                       	jae	101 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x22f>
100001d1a: 85 f6                       	testl	%esi, %esi
100001d1c: 74 f2                       	je	-14 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x1c0>
100001d1e: 49 63 d0                    	movslq	%r8d, %rdx
100001d21: 31 ff                       	xorl	%edi, %edi
100001d23: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100001d2d: 0f 1f 00                    	nopl	(%rax)
100001d30: 41 8d 34 39                 	leal	(%r9,%rdi), %esi
100001d34: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
100001d38: 48 8b 1b                    	movq	(%rbx), %rbx
100001d3b: 48 0f af da                 	imulq	%rdx, %rbx
100001d3f: 48 03 9d 50 ff ff ff        	addq	-176(%rbp), %rbx
100001d46: 48 63 ff                    	movslq	%edi, %rdi
100001d49: 0f b6 1c 1f                 	movzbl	(%rdi,%rbx), %ebx
100001d4d: d0 eb                       	shrb	%bl
100001d4f: 49 8d 45 28                 	leaq	40(%r13), %rax
100001d53: 49 8d 4d 30                 	leaq	48(%r13), %rcx
100001d57: 41 80 7d 24 00              	cmpb	$0, 36(%r13)
100001d5c: 48 0f 44 c8                 	cmoveq	%rax, %rcx
100001d60: 48 8b 01                    	movq	(%rcx), %rax
100001d63: 88 1c 30                    	movb	%bl, (%rax,%rsi)
100001d66: ff c7                       	incl	%edi
100001d68: 41 8b 75 10                 	movl	16(%r13), %esi
100001d6c: 39 f7                       	cmpl	%esi, %edi
100001d6e: 72 c0                       	jb	-64 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x1e0>
100001d70: 41 8b 55 0c                 	movl	12(%r13), %edx
100001d74: 41 01 f9                    	addl	%edi, %r9d
100001d77: 41 ff c0                    	incl	%r8d
100001d7a: 41 39 d0                    	cmpl	%edx, %r8d
100001d7d: 72 9b                       	jb	-101 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x1ca>
100001d7f: 49 8b 45 00                 	movq	(%r13), %rax
100001d83: 4c 89 ef                    	movq	%r13, %rdi
100001d86: ff 50 10                    	callq	*16(%rax)
100001d89: 41 8b 45 18                 	movl	24(%r13), %eax
100001d8d: 41 8b 4d 1c                 	movl	28(%r13), %ecx
100001d91: 41 c7 06 00 00 ff 42        	movl	$1124007936, (%r14)
100001d98: 49 8d 56 08                 	leaq	8(%r14), %rdx
100001d9c: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001da0: c4 c1 7c 11 46 04           	vmovups	%ymm0, 4(%r14)
100001da6: c4 c1 7c 11 46 20           	vmovups	%ymm0, 32(%r14)
100001dac: 49 89 56 40                 	movq	%rdx, 64(%r14)
100001db0: 49 8d 56 50                 	leaq	80(%r14), %rdx
100001db4: 49 89 56 48                 	movq	%rdx, 72(%r14)
100001db8: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100001dbc: c4 c1 78 11 46 50           	vmovups	%xmm0, 80(%r14)
100001dc2: 89 4d b8                    	movl	%ecx, -72(%rbp)
100001dc5: 89 45 bc                    	movl	%eax, -68(%rbp)
100001dc8: 48 8d 55 b8                 	leaq	-72(%rbp), %rdx
100001dcc: 4c 89 f7                    	movq	%r14, %rdi
100001dcf: be 02 00 00 00              	movl	$2, %esi
100001dd4: 31 c9                       	xorl	%ecx, %ecx
100001dd6: c5 f8 77                    	vzeroupper
100001dd9: e8 5e 50 00 00              	callq	20574 <dyld_stub_binder+0x100006e3c>
100001dde: 41 8b 45 18                 	movl	24(%r13), %eax
100001de2: 41 83 7d 14 01              	cmpl	$1, 20(%r13)
100001de7: 0f 85 d6 00 00 00           	jne	214 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x373>
100001ded: 85 c0                       	testl	%eax, %eax
100001def: 0f 84 d7 02 00 00           	je	727 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001df5: c5 fa 10 45 b0              	vmovss	-80(%rbp), %xmm0
100001dfa: c5 fa 59 05 c6 52 00 00     	vmulss	21190(%rip), %xmm0, %xmm0
100001e02: 41 8b 7d 1c                 	movl	28(%r13), %edi
100001e06: 45 31 c0                    	xorl	%r8d, %r8d
100001e09: 31 f6                       	xorl	%esi, %esi
100001e0b: 85 ff                       	testl	%edi, %edi
100001e0d: 75 27                       	jne	39 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2e6>
100001e0f: e9 9c 00 00 00              	jmp	156 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x360>
100001e14: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100001e1e: 66 90                       	nop
100001e20: 41 8b 45 18                 	movl	24(%r13), %eax
100001e24: 01 d6                       	addl	%edx, %esi
100001e26: 41 ff c0                    	incl	%r8d
100001e29: 41 39 c0                    	cmpl	%eax, %r8d
100001e2c: 0f 83 9a 02 00 00           	jae	666 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001e32: 85 ff                       	testl	%edi, %edi
100001e34: 74 7a                       	je	122 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x360>
100001e36: 49 63 c0                    	movslq	%r8d, %rax
100001e39: 31 d2                       	xorl	%edx, %edx
100001e3b: eb 41                       	jmp	65 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x32e>
100001e3d: 0f 1f 00                    	nopl	(%rax)
100001e40: 40 0f be cf                 	movsbl	%dil, %ecx
100001e44: c5 ea 2a c9                 	vcvtsi2ss	%ecx, %xmm2, %xmm1
100001e48: 49 8b 7e 48                 	movq	72(%r14), %rdi
100001e4c: 48 8b 3f                    	movq	(%rdi), %rdi
100001e4f: 48 0f af f8                 	imulq	%rax, %rdi
100001e53: 49 03 7e 10                 	addq	16(%r14), %rdi
100001e57: 48 63 d2                    	movslq	%edx, %rdx
100001e5a: 88 0c 3a                    	movb	%cl, (%rdx,%rdi)
100001e5d: 49 8b 4e 48                 	movq	72(%r14), %rcx
100001e61: 48 8b 09                    	movq	(%rcx), %rcx
100001e64: 48 0f af c8                 	imulq	%rax, %rcx
100001e68: 49 03 4e 10                 	addq	16(%r14), %rcx
100001e6c: c5 f8 2e c8                 	vucomiss	%xmm0, %xmm1
100001e70: 0f 97 04 0a                 	seta	(%rdx,%rcx)
100001e74: ff c2                       	incl	%edx
100001e76: 41 8b 7d 1c                 	movl	28(%r13), %edi
100001e7a: 39 fa                       	cmpl	%edi, %edx
100001e7c: 73 a2                       	jae	-94 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2d0>
100001e7e: 8d 3c 16                    	leal	(%rsi,%rdx), %edi
100001e81: 49 8d 5d 30                 	leaq	48(%r13), %rbx
100001e85: 49 8d 4d 28                 	leaq	40(%r13), %rcx
100001e89: 41 80 7d 24 00              	cmpb	$0, 36(%r13)
100001e8e: 48 0f 44 cb                 	cmoveq	%rbx, %rcx
100001e92: 48 8b 09                    	movq	(%rcx), %rcx
100001e95: 0f b6 3c 39                 	movzbl	(%rcx,%rdi), %edi
100001e99: 40 84 ff                    	testb	%dil, %dil
100001e9c: 79 a2                       	jns	-94 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2f0>
100001e9e: 31 ff                       	xorl	%edi, %edi
100001ea0: eb 9e                       	jmp	-98 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2f0>
100001ea2: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100001eac: 0f 1f 40 00                 	nopl	(%rax)
100001eb0: 31 ff                       	xorl	%edi, %edi
100001eb2: 41 ff c0                    	incl	%r8d
100001eb5: 41 39 c0                    	cmpl	%eax, %r8d
100001eb8: 0f 82 74 ff ff ff           	jb	-140 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x2e2>
100001ebe: e9 09 02 00 00              	jmp	521 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001ec3: 85 c0                       	testl	%eax, %eax
100001ec5: 0f 84 01 02 00 00           	je	513 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001ecb: c5 fa 10 45 b0              	vmovss	-80(%rbp), %xmm0
100001ed0: c5 fa 59 05 f0 51 00 00     	vmulss	20976(%rip), %xmm0, %xmm0
100001ed8: 41 8b 4d 1c                 	movl	28(%r13), %ecx
100001edc: 45 31 c0                    	xorl	%r8d, %r8d
100001edf: 31 f6                       	xorl	%esi, %esi
100001ee1: 45 31 d2                    	xorl	%r10d, %r10d
100001ee4: 85 c9                       	testl	%ecx, %ecx
100001ee6: 75 21                       	jne	33 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3b9>
100001ee8: e9 d3 01 00 00              	jmp	467 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x570>
100001eed: 0f 1f 00                    	nopl	(%rax)
100001ef0: 41 8b 45 18                 	movl	24(%r13), %eax
100001ef4: 8b 75 ac                    	movl	-84(%rbp), %esi
100001ef7: ff c6                       	incl	%esi
100001ef9: 39 c6                       	cmpl	%eax, %esi
100001efb: 0f 83 cb 01 00 00           	jae	459 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x57c>
100001f01: 85 c9                       	testl	%ecx, %ecx
100001f03: 0f 84 b7 01 00 00           	je	439 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x570>
100001f09: 89 75 ac                    	movl	%esi, -84(%rbp)
100001f0c: 48 63 d6                    	movslq	%esi, %rdx
100001f0f: 45 31 db                    	xorl	%r11d, %r11d
100001f12: 48 89 55 b0                 	movq	%rdx, -80(%rbp)
100001f16: 45 8b 7d 14                 	movl	20(%r13), %r15d
100001f1a: 45 85 ff                    	testl	%r15d, %r15d
100001f1d: 75 3f                       	jne	63 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x40e>
100001f1f: 90                          	nop
100001f20: b0 81                       	movb	$-127, %al
100001f22: 31 ff                       	xorl	%edi, %edi
100001f24: 0f be c0                    	movsbl	%al, %eax
100001f27: c5 ea 2a c8                 	vcvtsi2ss	%eax, %xmm2, %xmm1
100001f2b: c5 f8 2e c8                 	vucomiss	%xmm0, %xmm1
100001f2f: 41 0f 46 f8                 	cmovbel	%r8d, %edi
100001f33: 49 8b 46 48                 	movq	72(%r14), %rax
100001f37: 48 8b 00                    	movq	(%rax), %rax
100001f3a: 48 0f af c2                 	imulq	%rdx, %rax
100001f3e: 49 03 46 10                 	addq	16(%r14), %rax
100001f42: 4d 63 db                    	movslq	%r11d, %r11
100001f45: 41 88 3c 03                 	movb	%dil, (%r11,%rax)
100001f49: 41 ff c3                    	incl	%r11d
100001f4c: 41 8b 4d 1c                 	movl	28(%r13), %ecx
100001f50: 41 39 cb                    	cmpl	%ecx, %r11d
100001f53: 73 9b                       	jae	-101 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3a0>
100001f55: 45 8b 7d 14                 	movl	20(%r13), %r15d
100001f59: 45 85 ff                    	testl	%r15d, %r15d
100001f5c: 74 c2                       	je	-62 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3d0>
100001f5e: 49 8d 45 30                 	leaq	48(%r13), %rax
100001f62: 41 80 7d 24 00              	cmpb	$0, 36(%r13)
100001f67: 49 8d 4d 28                 	leaq	40(%r13), %rcx
100001f6b: 48 0f 44 c8                 	cmoveq	%rax, %rcx
100001f6f: 4c 8b 09                    	movq	(%rcx), %r9
100001f72: 41 8d 47 ff                 	leal	-1(%r15), %eax
100001f76: 45 89 fc                    	movl	%r15d, %r12d
100001f79: 41 83 e4 07                 	andl	$7, %r12d
100001f7d: 83 f8 07                    	cmpl	$7, %eax
100001f80: 73 1e                       	jae	30 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x450>
100001f82: b0 81                       	movb	$-127, %al
100001f84: 31 db                       	xorl	%ebx, %ebx
100001f86: 31 ff                       	xorl	%edi, %edi
100001f88: 45 85 e4                    	testl	%r12d, %r12d
100001f8b: 0f 85 ff 00 00 00           	jne	255 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x540>
100001f91: eb 91                       	jmp	-111 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3d4>
100001f93: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100001f9d: 0f 1f 00                    	nopl	(%rax)
100001fa0: 45 29 e7                    	subl	%r12d, %r15d
100001fa3: b0 81                       	movb	$-127, %al
100001fa5: 31 db                       	xorl	%ebx, %ebx
100001fa7: 31 ff                       	xorl	%edi, %edi
100001fa9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100001fb0: 41 8d 34 1a                 	leal	(%r10,%rbx), %esi
100001fb4: 41 0f b6 34 31              	movzbl	(%r9,%rsi), %esi
100001fb9: 45 8d 04 1a                 	leal	(%r10,%rbx), %r8d
100001fbd: 41 83 c0 01                 	addl	$1, %r8d
100001fc1: 40 38 c6                    	cmpb	%al, %sil
100001fc4: 0f 4f fb                    	cmovgl	%ebx, %edi
100001fc7: 0f b6 c0                    	movzbl	%al, %eax
100001fca: 0f 4d c6                    	cmovgel	%esi, %eax
100001fcd: 43 0f b6 34 01              	movzbl	(%r9,%r8), %esi
100001fd2: 41 8d 0c 1a                 	leal	(%r10,%rbx), %ecx
100001fd6: 83 c1 02                    	addl	$2, %ecx
100001fd9: 8d 53 01                    	leal	1(%rbx), %edx
100001fdc: 40 38 c6                    	cmpb	%al, %sil
100001fdf: 0f 4e d7                    	cmovlel	%edi, %edx
100001fe2: 0f 4d c6                    	cmovgel	%esi, %eax
100001fe5: 41 0f b6 0c 09              	movzbl	(%r9,%rcx), %ecx
100001fea: 41 8d 34 1a                 	leal	(%r10,%rbx), %esi
100001fee: 83 c6 03                    	addl	$3, %esi
100001ff1: 8d 7b 02                    	leal	2(%rbx), %edi
100001ff4: 38 c1                       	cmpb	%al, %cl
100001ff6: 0f 4e fa                    	cmovlel	%edx, %edi
100001ff9: 0f 4d c1                    	cmovgel	%ecx, %eax
100001ffc: 41 0f b6 0c 31              	movzbl	(%r9,%rsi), %ecx
100002001: 41 8d 14 1a                 	leal	(%r10,%rbx), %edx
100002005: 83 c2 04                    	addl	$4, %edx
100002008: 8d 73 03                    	leal	3(%rbx), %esi
10000200b: 38 c1                       	cmpb	%al, %cl
10000200d: 0f 4e f7                    	cmovlel	%edi, %esi
100002010: 0f 4d c1                    	cmovgel	%ecx, %eax
100002013: 41 0f b6 0c 11              	movzbl	(%r9,%rdx), %ecx
100002018: 41 8d 14 1a                 	leal	(%r10,%rbx), %edx
10000201c: 83 c2 05                    	addl	$5, %edx
10000201f: 8d 7b 04                    	leal	4(%rbx), %edi
100002022: 38 c1                       	cmpb	%al, %cl
100002024: 0f 4e fe                    	cmovlel	%esi, %edi
100002027: 0f 4d c1                    	cmovgel	%ecx, %eax
10000202a: 41 0f b6 0c 11              	movzbl	(%r9,%rdx), %ecx
10000202f: 41 8d 14 1a                 	leal	(%r10,%rbx), %edx
100002033: 83 c2 06                    	addl	$6, %edx
100002036: 8d 73 05                    	leal	5(%rbx), %esi
100002039: 38 c1                       	cmpb	%al, %cl
10000203b: 0f 4e f7                    	cmovlel	%edi, %esi
10000203e: 0f 4d c1                    	cmovgel	%ecx, %eax
100002041: 41 0f b6 0c 11              	movzbl	(%r9,%rdx), %ecx
100002046: 41 8d 3c 1a                 	leal	(%r10,%rbx), %edi
10000204a: 83 c7 07                    	addl	$7, %edi
10000204d: 8d 53 06                    	leal	6(%rbx), %edx
100002050: 38 c1                       	cmpb	%al, %cl
100002052: 0f 4e d6                    	cmovlel	%esi, %edx
100002055: 0f 4d c1                    	cmovgel	%ecx, %eax
100002058: 41 0f b6 0c 39              	movzbl	(%r9,%rdi), %ecx
10000205d: 8d 7b 07                    	leal	7(%rbx), %edi
100002060: 38 c1                       	cmpb	%al, %cl
100002062: 0f 4e fa                    	cmovlel	%edx, %edi
100002065: 0f 4d c1                    	cmovgel	%ecx, %eax
100002068: 83 c3 08                    	addl	$8, %ebx
10000206b: 41 39 df                    	cmpl	%ebx, %r15d
10000206e: 0f 85 3c ff ff ff           	jne	-196 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x460>
100002074: 41 01 da                    	addl	%ebx, %r10d
100002077: 45 31 c0                    	xorl	%r8d, %r8d
10000207a: 48 8b 55 b0                 	movq	-80(%rbp), %rdx
10000207e: 45 85 e4                    	testl	%r12d, %r12d
100002081: 0f 84 9d fe ff ff           	je	-355 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3d4>
100002087: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
100002090: 44 89 d6                    	movl	%r10d, %esi
100002093: 41 0f b6 34 31              	movzbl	(%r9,%rsi), %esi
100002098: 41 ff c2                    	incl	%r10d
10000209b: 40 38 c6                    	cmpb	%al, %sil
10000209e: 0f 4f fb                    	cmovgl	%ebx, %edi
1000020a1: 0f b6 c0                    	movzbl	%al, %eax
1000020a4: 0f 4d c6                    	cmovgel	%esi, %eax
1000020a7: ff c3                       	incl	%ebx
1000020a9: 41 ff cc                    	decl	%r12d
1000020ac: 75 e2                       	jne	-30 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x540>
1000020ae: e9 71 fe ff ff              	jmp	-399 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3d4>
1000020b3: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000020bd: 0f 1f 00                    	nopl	(%rax)
1000020c0: 31 c9                       	xorl	%ecx, %ecx
1000020c2: ff c6                       	incl	%esi
1000020c4: 39 c6                       	cmpl	%eax, %esi
1000020c6: 0f 82 35 fe ff ff           	jb	-459 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x3b1>
1000020cc: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
1000020d3: 48 85 c0                    	testq	%rax, %rax
1000020d6: 74 12                       	je	18 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x59a>
1000020d8: f0                          	lock
1000020d9: ff 48 14                    	decl	20(%rax)
1000020dc: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x59a>
1000020de: 48 8d bd 40 ff ff ff        	leaq	-192(%rbp), %rdi
1000020e5: e8 4c 4d 00 00              	callq	19788 <dyld_stub_binder+0x100006e36>
1000020ea: 48 c7 85 78 ff ff ff 00 00 00 00    	movq	$0, -136(%rbp)
1000020f5: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000020f9: c5 fc 11 85 50 ff ff ff     	vmovups	%ymm0, -176(%rbp)
100002101: 83 bd 44 ff ff ff 00        	cmpl	$0, -188(%rbp)
100002108: 7e 1c                       	jle	28 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x5d6>
10000210a: 48 8b 45 80                 	movq	-128(%rbp), %rax
10000210e: 31 c9                       	xorl	%ecx, %ecx
100002110: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002117: 48 ff c1                    	incq	%rcx
10000211a: 48 63 95 44 ff ff ff        	movslq	-188(%rbp), %rdx
100002121: 48 39 d1                    	cmpq	%rdx, %rcx
100002124: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x5c0>
100002126: 48 8b 7d 88                 	movq	-120(%rbp), %rdi
10000212a: 48 8d 45 90                 	leaq	-112(%rbp), %rax
10000212e: 48 39 c7                    	cmpq	%rax, %rdi
100002131: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x5eb>
100002133: c5 f8 77                    	vzeroupper
100002136: e8 31 4d 00 00              	callq	19761 <dyld_stub_binder+0x100006e6c>
10000213b: 48 8b 85 18 ff ff ff        	movq	-232(%rbp), %rax
100002142: 48 85 c0                    	testq	%rax, %rax
100002145: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x60c>
100002147: f0                          	lock
100002148: ff 48 14                    	decl	20(%rax)
10000214b: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x60c>
10000214d: 48 8d bd e0 fe ff ff        	leaq	-288(%rbp), %rdi
100002154: c5 f8 77                    	vzeroupper
100002157: e8 da 4c 00 00              	callq	19674 <dyld_stub_binder+0x100006e36>
10000215c: 48 c7 85 18 ff ff ff 00 00 00 00    	movq	$0, -232(%rbp)
100002167: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000216b: c5 fc 11 85 f0 fe ff ff     	vmovups	%ymm0, -272(%rbp)
100002173: 83 bd e4 fe ff ff 00        	cmpl	$0, -284(%rbp)
10000217a: 7e 2a                       	jle	42 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x656>
10000217c: 48 8b 85 20 ff ff ff        	movq	-224(%rbp), %rax
100002183: 31 c9                       	xorl	%ecx, %ecx
100002185: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000218f: 90                          	nop
100002190: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002197: 48 ff c1                    	incq	%rcx
10000219a: 48 63 95 e4 fe ff ff        	movslq	-284(%rbp), %rdx
1000021a1: 48 39 d1                    	cmpq	%rdx, %rcx
1000021a4: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x640>
1000021a6: 48 8b bd 28 ff ff ff        	movq	-216(%rbp), %rdi
1000021ad: 48 8d 85 30 ff ff ff        	leaq	-208(%rbp), %rax
1000021b4: 48 39 c7                    	cmpq	%rax, %rdi
1000021b7: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x671>
1000021b9: c5 f8 77                    	vzeroupper
1000021bc: e8 ab 4c 00 00              	callq	19627 <dyld_stub_binder+0x100006e6c>
1000021c1: 48 8b 05 98 6e 00 00        	movq	28312(%rip), %rax
1000021c8: 48 8b 00                    	movq	(%rax), %rax
1000021cb: 48 3b 45 d0                 	cmpq	-48(%rbp), %rax
1000021cf: 75 18                       	jne	24 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x699>
1000021d1: 4c 89 f0                    	movq	%r14, %rax
1000021d4: 48 81 c4 18 01 00 00        	addq	$280, %rsp
1000021db: 5b                          	popq	%rbx
1000021dc: 41 5c                       	popq	%r12
1000021de: 41 5d                       	popq	%r13
1000021e0: 41 5e                       	popq	%r14
1000021e2: 41 5f                       	popq	%r15
1000021e4: 5d                          	popq	%rbp
1000021e5: c5 f8 77                    	vzeroupper
1000021e8: c3                          	retq
1000021e9: c5 f8 77                    	vzeroupper
1000021ec: e8 ff 4c 00 00              	callq	19711 <dyld_stub_binder+0x100006ef0>
1000021f1: 48 89 c7                    	movq	%rax, %rdi
1000021f4: e8 f7 16 00 00              	callq	5879 <_main+0x15b0>
1000021f9: 48 89 c7                    	movq	%rax, %rdi
1000021fc: e8 ef 16 00 00              	callq	5871 <_main+0x15b0>
100002201: 48 89 c3                    	movq	%rax, %rbx
100002204: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
10000220b: 48 85 c0                    	testq	%rax, %rax
10000220e: 75 2b                       	jne	43 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6eb>
100002210: eb 3b                       	jmp	59 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6fd>
100002212: eb 00                       	jmp	0 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6c4>
100002214: 48 89 c3                    	movq	%rax, %rbx
100002217: 48 8b 85 18 ff ff ff        	movq	-232(%rbp), %rax
10000221e: 48 85 c0                    	testq	%rax, %rax
100002221: 0f 85 83 00 00 00           	jne	131 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x75a>
100002227: e9 93 00 00 00              	jmp	147 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x76f>
10000222c: 48 89 c3                    	movq	%rax, %rbx
10000222f: 48 8b 85 78 ff ff ff        	movq	-136(%rbp), %rax
100002236: 48 85 c0                    	testq	%rax, %rax
100002239: 74 12                       	je	18 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6fd>
10000223b: f0                          	lock
10000223c: ff 48 14                    	decl	20(%rax)
10000223f: 75 0c                       	jne	12 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x6fd>
100002241: 48 8d bd 40 ff ff ff        	leaq	-192(%rbp), %rdi
100002248: e8 e9 4b 00 00              	callq	19433 <dyld_stub_binder+0x100006e36>
10000224d: 48 c7 85 78 ff ff ff 00 00 00 00    	movq	$0, -136(%rbp)
100002258: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
10000225c: c5 fc 11 85 50 ff ff ff     	vmovups	%ymm0, -176(%rbp)
100002264: 83 bd 44 ff ff ff 00        	cmpl	$0, -188(%rbp)
10000226b: 7e 1c                       	jle	28 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x739>
10000226d: 48 8b 45 80                 	movq	-128(%rbp), %rax
100002271: 31 c9                       	xorl	%ecx, %ecx
100002273: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
10000227a: 48 ff c1                    	incq	%rcx
10000227d: 48 63 95 44 ff ff ff        	movslq	-188(%rbp), %rdx
100002284: 48 39 d1                    	cmpq	%rdx, %rcx
100002287: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x723>
100002289: 48 8b 7d 88                 	movq	-120(%rbp), %rdi
10000228d: 48 8d 45 90                 	leaq	-112(%rbp), %rax
100002291: 48 39 c7                    	cmpq	%rax, %rdi
100002294: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x74e>
100002296: c5 f8 77                    	vzeroupper
100002299: e8 ce 4b 00 00              	callq	19406 <dyld_stub_binder+0x100006e6c>
10000229e: 48 8b 85 18 ff ff ff        	movq	-232(%rbp), %rax
1000022a5: 48 85 c0                    	testq	%rax, %rax
1000022a8: 74 15                       	je	21 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x76f>
1000022aa: f0                          	lock
1000022ab: ff 48 14                    	decl	20(%rax)
1000022ae: 75 0f                       	jne	15 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x76f>
1000022b0: 48 8d bd e0 fe ff ff        	leaq	-288(%rbp), %rdi
1000022b7: c5 f8 77                    	vzeroupper
1000022ba: e8 77 4b 00 00              	callq	19319 <dyld_stub_binder+0x100006e36>
1000022bf: 48 c7 85 18 ff ff ff 00 00 00 00    	movq	$0, -232(%rbp)
1000022ca: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000022ce: c5 fc 11 85 f0 fe ff ff     	vmovups	%ymm0, -272(%rbp)
1000022d6: 83 bd e4 fe ff ff 00        	cmpl	$0, -284(%rbp)
1000022dd: 7e 27                       	jle	39 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x7b6>
1000022df: 48 8b 85 20 ff ff ff        	movq	-224(%rbp), %rax
1000022e6: 31 c9                       	xorl	%ecx, %ecx
1000022e8: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
1000022f0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000022f7: 48 ff c1                    	incq	%rcx
1000022fa: 48 63 95 e4 fe ff ff        	movslq	-284(%rbp), %rdx
100002301: 48 39 d1                    	cmpq	%rdx, %rcx
100002304: 7c ea                       	jl	-22 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x7a0>
100002306: 48 8b bd 28 ff ff ff        	movq	-216(%rbp), %rdi
10000230d: 48 8d 85 30 ff ff ff        	leaq	-208(%rbp), %rax
100002314: 48 39 c7                    	cmpq	%rax, %rdi
100002317: 74 08                       	je	8 <__Z14get_predictionRN2cv3MatER11LineNetworkf+0x7d1>
100002319: c5 f8 77                    	vzeroupper
10000231c: e8 4b 4b 00 00              	callq	19275 <dyld_stub_binder+0x100006e6c>
100002321: 48 89 df                    	movq	%rbx, %rdi
100002324: c5 f8 77                    	vzeroupper
100002327: e8 ec 4a 00 00              	callq	19180 <dyld_stub_binder+0x100006e18>
10000232c: 0f 0b                       	ud2
10000232e: 48 89 c7                    	movq	%rax, %rdi
100002331: e8 ba 15 00 00              	callq	5562 <_main+0x15b0>
100002336: 48 89 c7                    	movq	%rax, %rdi
100002339: e8 b2 15 00 00              	callq	5554 <_main+0x15b0>
10000233e: 66 90                       	nop

0000000100002340 _main:
100002340: 55                          	pushq	%rbp
100002341: 48 89 e5                    	movq	%rsp, %rbp
100002344: 41 57                       	pushq	%r15
100002346: 41 56                       	pushq	%r14
100002348: 41 55                       	pushq	%r13
10000234a: 41 54                       	pushq	%r12
10000234c: 53                          	pushq	%rbx
10000234d: 48 83 e4 e0                 	andq	$-32, %rsp
100002351: 48 81 ec 00 04 00 00        	subq	$1024, %rsp
100002358: 48 8b 05 01 6d 00 00        	movq	27905(%rip), %rax
10000235f: 48 8b 00                    	movq	(%rax), %rax
100002362: 48 89 84 24 e0 03 00 00     	movq	%rax, 992(%rsp)
10000236a: 48 8d bc 24 d8 01 00 00     	leaq	472(%rsp), %rdi
100002372: e8 29 1d 00 00              	callq	7465 <__ZN11LineNetworkC1Ev>
100002377: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000237b: c5 f9 7f 84 24 60 02 00 00  	vmovdqa	%xmm0, 608(%rsp)
100002384: 48 c7 84 24 70 02 00 00 00 00 00 00 	movq	$0, 624(%rsp)
100002390: bf 30 00 00 00              	movl	$48, %edi
100002395: e8 44 4b 00 00              	callq	19268 <dyld_stub_binder+0x100006ede>
10000239a: 48 89 84 24 70 02 00 00     	movq	%rax, 624(%rsp)
1000023a2: c5 f8 28 05 46 4d 00 00     	vmovaps	19782(%rip), %xmm0
1000023aa: c5 f8 29 84 24 60 02 00 00  	vmovaps	%xmm0, 608(%rsp)
1000023b3: c5 fe 6f 05 21 6b 00 00     	vmovdqu	27425(%rip), %ymm0
1000023bb: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
1000023bf: 48 b9 69 64 65 6f 2e 6d 70 34       	movabsq	$3778640133568685161, %rcx
1000023c9: 48 89 48 20                 	movq	%rcx, 32(%rax)
1000023cd: c6 40 28 00                 	movb	$0, 40(%rax)
1000023d1: 48 8d bc 24 10 02 00 00     	leaq	528(%rsp), %rdi
1000023d9: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
1000023e1: 31 d2                       	xorl	%edx, %edx
1000023e3: c5 f8 77                    	vzeroupper
1000023e6: e8 39 4a 00 00              	callq	19001 <dyld_stub_binder+0x100006e24>
1000023eb: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
1000023f3: 74 0d                       	je	13 <_main+0xc2>
1000023f5: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
1000023fd: e8 d0 4a 00 00              	callq	19152 <dyld_stub_binder+0x100006ed2>
100002402: 4c 8d 6c 24 68              	leaq	104(%rsp), %r13
100002407: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000240b: c5 f9 d6 44 24 78           	vmovq	%xmm0, 120(%rsp)
100002411: 48 8d 9c 24 10 02 00 00     	leaq	528(%rsp), %rbx
100002419: 4c 8d b4 24 c0 03 00 00     	leaq	960(%rsp), %r14
100002421: 4c 8d a4 24 c0 01 00 00     	leaq	448(%rsp), %r12
100002429: eb 0e                       	jmp	14 <_main+0xf9>
10000242b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100002430: 45 85 ff                    	testl	%r15d, %r15d
100002433: 0f 85 d1 0f 00 00           	jne	4049 <_main+0x10ca>
100002439: 48 89 df                    	movq	%rbx, %rdi
10000243c: c5 f8 77                    	vzeroupper
10000243f: e8 34 4a 00 00              	callq	18996 <dyld_stub_binder+0x100006e78>
100002444: 84 c0                       	testb	%al, %al
100002446: 0f 84 be 0f 00 00           	je	4030 <_main+0x10ca>
10000244c: c7 44 24 18 00 00 ff 42     	movl	$1124007936, 24(%rsp)
100002454: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002458: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
10000245d: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
100002462: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100002466: 48 8d 44 24 20              	leaq	32(%rsp), %rax
10000246b: 48 89 44 24 58              	movq	%rax, 88(%rsp)
100002470: 4c 89 6c 24 60              	movq	%r13, 96(%rsp)
100002475: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002479: c4 c1 7a 7f 45 00           	vmovdqu	%xmm0, (%r13)
10000247f: 48 89 df                    	movq	%rbx, %rdi
100002482: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100002487: c5 f8 77                    	vzeroupper
10000248a: e8 a1 49 00 00              	callq	18849 <dyld_stub_binder+0x100006e30>
10000248f: 41 bf 03 00 00 00           	movl	$3, %r15d
100002495: 48 83 7c 24 28 00           	cmpq	$0, 40(%rsp)
10000249b: 0f 84 8f 08 00 00           	je	2191 <_main+0x9f0>
1000024a1: 8b 44 24 1c                 	movl	28(%rsp), %eax
1000024a5: 83 f8 03                    	cmpl	$3, %eax
1000024a8: 0f 8d 62 03 00 00           	jge	866 <_main+0x4d0>
1000024ae: 48 63 4c 24 20              	movslq	32(%rsp), %rcx
1000024b3: 48 63 74 24 24              	movslq	36(%rsp), %rsi
1000024b8: 48 0f af f1                 	imulq	%rcx, %rsi
1000024bc: 85 c0                       	testl	%eax, %eax
1000024be: 0f 84 6c 08 00 00           	je	2156 <_main+0x9f0>
1000024c4: 48 85 f6                    	testq	%rsi, %rsi
1000024c7: 0f 84 63 08 00 00           	je	2147 <_main+0x9f0>
1000024cd: bf 19 00 00 00              	movl	$25, %edi
1000024d2: c5 f8 77                    	vzeroupper
1000024d5: e8 86 49 00 00              	callq	18822 <dyld_stub_binder+0x100006e60>
1000024da: 3c 1b                       	cmpb	$27, %al
1000024dc: 0f 84 4e 08 00 00           	je	2126 <_main+0x9f0>
1000024e2: e8 c1 49 00 00              	callq	18881 <dyld_stub_binder+0x100006ea8>
1000024e7: 49 89 c5                    	movq	%rax, %r13
1000024ea: 48 8d 9c 24 e0 00 00 00     	leaq	224(%rsp), %rbx
1000024f2: 48 89 df                    	movq	%rbx, %rdi
1000024f5: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
1000024fa: 48 8d 94 24 d8 01 00 00     	leaq	472(%rsp), %rdx
100002502: c5 f9 6e 05 c2 4b 00 00     	vmovd	19394(%rip), %xmm0
10000250a: e8 41 f6 ff ff              	callq	-2495 <__Z14get_predictionRN2cv3MatER11LineNetworkf>
10000250f: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
100002517: c5 fa 7e 05 71 4b 00 00     	vmovq	19313(%rip), %xmm0
10000251f: 48 89 de                    	movq	%rbx, %rsi
100002522: e8 4b 49 00 00              	callq	18763 <dyld_stub_binder+0x100006e72>
100002527: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
10000252f: 48 85 c0                    	testq	%rax, %rax
100002532: 74 0e                       	je	14 <_main+0x202>
100002534: f0                          	lock
100002535: ff 48 14                    	decl	20(%rax)
100002538: 75 08                       	jne	8 <_main+0x202>
10000253a: 48 89 df                    	movq	%rbx, %rdi
10000253d: e8 f4 48 00 00              	callq	18676 <dyld_stub_binder+0x100006e36>
100002542: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
10000254e: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100002556: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
10000255a: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
10000255e: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
100002566: 7e 2f                       	jle	47 <_main+0x257>
100002568: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100002570: 31 c9                       	xorl	%ecx, %ecx
100002572: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000257c: 0f 1f 40 00                 	nopl	(%rax)
100002580: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002587: 48 ff c1                    	incq	%rcx
10000258a: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
100002592: 48 39 d1                    	cmpq	%rdx, %rcx
100002595: 7c e9                       	jl	-23 <_main+0x240>
100002597: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
10000259f: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
1000025a7: 48 39 c7                    	cmpq	%rax, %rdi
1000025aa: 74 08                       	je	8 <_main+0x274>
1000025ac: c5 f8 77                    	vzeroupper
1000025af: e8 b8 48 00 00              	callq	18616 <dyld_stub_binder+0x100006e6c>
1000025b4: c5 f8 77                    	vzeroupper
1000025b7: e8 ec 48 00 00              	callq	18668 <dyld_stub_binder+0x100006ea8>
1000025bc: 49 89 c7                    	movq	%rax, %r15
1000025bf: c7 84 24 e0 00 00 00 00 00 ff 42    	movl	$1124007936, 224(%rsp)
1000025ca: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
1000025d2: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000025d6: c5 fe 7f 40 f4              	vmovdqu	%ymm0, -12(%rax)
1000025db: c5 fe 7f 40 10              	vmovdqu	%ymm0, 16(%rax)
1000025e0: 48 8b 44 24 20              	movq	32(%rsp), %rax
1000025e5: 48 8d 8c 24 e8 00 00 00     	leaq	232(%rsp), %rcx
1000025ed: 48 89 8c 24 20 01 00 00     	movq	%rcx, 288(%rsp)
1000025f5: 48 8d 8c 24 30 01 00 00     	leaq	304(%rsp), %rcx
1000025fd: 48 89 8c 24 28 01 00 00     	movq	%rcx, 296(%rsp)
100002605: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002609: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
10000260d: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
100002615: 48 89 df                    	movq	%rbx, %rdi
100002618: be 02 00 00 00              	movl	$2, %esi
10000261d: 4c 89 f2                    	movq	%r14, %rdx
100002620: 31 c9                       	xorl	%ecx, %ecx
100002622: c5 f8 77                    	vzeroupper
100002625: e8 12 48 00 00              	callq	18450 <dyld_stub_binder+0x100006e3c>
10000262a: 48 8d 9c 24 80 00 00 00     	leaq	128(%rsp), %rbx
100002632: 48 89 df                    	movq	%rbx, %rdi
100002635: 48 8d b4 24 60 02 00 00     	leaq	608(%rsp), %rsi
10000263d: e8 dc 47 00 00              	callq	18396 <dyld_stub_binder+0x100006e1e>
100002642: 48 c7 84 24 70 01 00 00 00 00 00 00 	movq	$0, 368(%rsp)
10000264e: c7 84 24 60 01 00 00 00 00 01 02    	movl	$33619968, 352(%rsp)
100002659: 48 8d 84 24 e0 00 00 00     	leaq	224(%rsp), %rax
100002661: 48 89 84 24 68 01 00 00     	movq	%rax, 360(%rsp)
100002669: 8b 44 24 20                 	movl	32(%rsp), %eax
10000266d: 8b 4c 24 24                 	movl	36(%rsp), %ecx
100002671: 89 8c 24 b0 01 00 00        	movl	%ecx, 432(%rsp)
100002678: 89 84 24 b4 01 00 00        	movl	%eax, 436(%rsp)
10000267f: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002683: c5 f1 ef c9                 	vpxor	%xmm1, %xmm1, %xmm1
100002687: 48 89 df                    	movq	%rbx, %rdi
10000268a: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100002692: 48 8d 94 24 b0 01 00 00     	leaq	432(%rsp), %rdx
10000269a: b9 01 00 00 00              	movl	$1, %ecx
10000269f: e8 b0 47 00 00              	callq	18352 <dyld_stub_binder+0x100006e54>
1000026a4: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000026a8: c5 fd 7f 84 24 60 01 00 00  	vmovdqa	%ymm0, 352(%rsp)
1000026b1: c7 84 24 80 00 00 00 00 00 ff 42    	movl	$1124007936, 128(%rsp)
1000026bc: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
1000026c4: c5 fe 7f 40 1c              	vmovdqu	%ymm0, 28(%rax)
1000026c9: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
1000026cd: 48 8b 44 24 20              	movq	32(%rsp), %rax
1000026d2: 48 8d 8c 24 88 00 00 00     	leaq	136(%rsp), %rcx
1000026da: 48 89 8c 24 c0 00 00 00     	movq	%rcx, 192(%rsp)
1000026e2: 48 8d 8c 24 d0 00 00 00     	leaq	208(%rsp), %rcx
1000026ea: 48 89 8c 24 c8 00 00 00     	movq	%rcx, 200(%rsp)
1000026f2: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
1000026f6: c5 fa 7f 01                 	vmovdqu	%xmm0, (%rcx)
1000026fa: 48 89 84 24 c0 03 00 00     	movq	%rax, 960(%rsp)
100002702: 48 89 df                    	movq	%rbx, %rdi
100002705: be 02 00 00 00              	movl	$2, %esi
10000270a: 4c 89 f2                    	movq	%r14, %rdx
10000270d: b9 10 00 00 00              	movl	$16, %ecx
100002712: c5 f8 77                    	vzeroupper
100002715: e8 22 47 00 00              	callq	18210 <dyld_stub_binder+0x100006e3c>
10000271a: 48 89 df                    	movq	%rbx, %rdi
10000271d: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100002725: e8 1e 47 00 00              	callq	18206 <dyld_stub_binder+0x100006e48>
10000272a: 48 8b 44 24 50              	movq	80(%rsp), %rax
10000272f: 48 85 c0                    	testq	%rax, %rax
100002732: 74 04                       	je	4 <_main+0x3f8>
100002734: f0                          	lock
100002735: ff 40 14                    	incl	20(%rax)
100002738: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100002740: 48 85 c0                    	testq	%rax, %rax
100002743: 74 13                       	je	19 <_main+0x418>
100002745: f0                          	lock
100002746: ff 48 14                    	decl	20(%rax)
100002749: 75 0d                       	jne	13 <_main+0x418>
10000274b: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
100002753: e8 de 46 00 00              	callq	18142 <dyld_stub_binder+0x100006e36>
100002758: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
100002764: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
10000276c: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002770: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100002775: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
10000277d: 0f 8e 2c 06 00 00           	jle	1580 <_main+0xa6f>
100002783: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
10000278b: 31 c9                       	xorl	%ecx, %ecx
10000278d: 0f 1f 00                    	nopl	(%rax)
100002790: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002797: 48 ff c1                    	incq	%rcx
10000279a: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
1000027a2: 48 39 d1                    	cmpq	%rdx, %rcx
1000027a5: 7c e9                       	jl	-23 <_main+0x450>
1000027a7: 8b 44 24 18                 	movl	24(%rsp), %eax
1000027ab: 89 84 24 80 00 00 00        	movl	%eax, 128(%rsp)
1000027b2: 83 fa 02                    	cmpl	$2, %edx
1000027b5: 0f 8f 0c 06 00 00           	jg	1548 <_main+0xa87>
1000027bb: 8b 44 24 1c                 	movl	28(%rsp), %eax
1000027bf: 83 f8 02                    	cmpl	$2, %eax
1000027c2: 0f 8f ff 05 00 00           	jg	1535 <_main+0xa87>
1000027c8: 89 84 24 84 00 00 00        	movl	%eax, 132(%rsp)
1000027cf: 8b 4c 24 20                 	movl	32(%rsp), %ecx
1000027d3: 8b 44 24 24                 	movl	36(%rsp), %eax
1000027d7: 89 8c 24 88 00 00 00        	movl	%ecx, 136(%rsp)
1000027de: 89 84 24 8c 00 00 00        	movl	%eax, 140(%rsp)
1000027e5: 48 8b 44 24 60              	movq	96(%rsp), %rax
1000027ea: 48 8b 10                    	movq	(%rax), %rdx
1000027ed: 48 8b b4 24 c8 00 00 00     	movq	200(%rsp), %rsi
1000027f5: 48 89 16                    	movq	%rdx, (%rsi)
1000027f8: 48 8b 40 08                 	movq	8(%rax), %rax
1000027fc: 48 89 46 08                 	movq	%rax, 8(%rsi)
100002800: e9 db 05 00 00              	jmp	1499 <_main+0xaa0>
100002805: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000280f: 90                          	nop
100002810: 48 8b 4c 24 58              	movq	88(%rsp), %rcx
100002815: 83 f8 0f                    	cmpl	$15, %eax
100002818: 77 0c                       	ja	12 <_main+0x4e6>
10000281a: be 01 00 00 00              	movl	$1, %esi
10000281f: 31 d2                       	xorl	%edx, %edx
100002821: e9 ea 04 00 00              	jmp	1258 <_main+0x9d0>
100002826: 89 c2                       	movl	%eax, %edx
100002828: 83 e2 f0                    	andl	$-16, %edx
10000282b: 48 8d 72 f0                 	leaq	-16(%rdx), %rsi
10000282f: 48 89 f7                    	movq	%rsi, %rdi
100002832: 48 c1 ef 04                 	shrq	$4, %rdi
100002836: 48 ff c7                    	incq	%rdi
100002839: 89 fb                       	movl	%edi, %ebx
10000283b: 83 e3 03                    	andl	$3, %ebx
10000283e: 48 83 fe 30                 	cmpq	$48, %rsi
100002842: 73 25                       	jae	37 <_main+0x529>
100002844: c4 e2 7d 59 05 3b 48 00 00  	vpbroadcastq	18491(%rip), %ymm0
10000284d: 31 ff                       	xorl	%edi, %edi
10000284f: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
100002853: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100002857: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
10000285b: 48 85 db                    	testq	%rbx, %rbx
10000285e: 0f 85 0e 03 00 00           	jne	782 <_main+0x832>
100002864: e9 d0 03 00 00              	jmp	976 <_main+0x8f9>
100002869: 48 89 de                    	movq	%rbx, %rsi
10000286c: 48 29 fe                    	subq	%rdi, %rsi
10000286f: c4 e2 7d 59 05 10 48 00 00  	vpbroadcastq	18448(%rip), %ymm0
100002878: 31 ff                       	xorl	%edi, %edi
10000287a: c5 fd 6f d8                 	vmovdqa	%ymm0, %ymm3
10000287e: c5 fd 6f d0                 	vmovdqa	%ymm0, %ymm2
100002882: c5 fd 6f c8                 	vmovdqa	%ymm0, %ymm1
100002886: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002890: c4 e2 7d 25 24 b9           	vpmovsxdq	(%rcx,%rdi,4), %ymm4
100002896: c4 e2 7d 25 6c b9 10        	vpmovsxdq	16(%rcx,%rdi,4), %ymm5
10000289d: c4 e2 7d 25 74 b9 20        	vpmovsxdq	32(%rcx,%rdi,4), %ymm6
1000028a4: c4 e2 7d 25 7c b9 30        	vpmovsxdq	48(%rcx,%rdi,4), %ymm7
1000028ab: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
1000028b0: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
1000028b4: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
1000028b9: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
1000028be: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
1000028c3: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
1000028c9: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
1000028cd: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
1000028d2: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
1000028d7: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
1000028db: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
1000028e0: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
1000028e5: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
1000028e9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000028ee: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
1000028f2: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000028f6: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
1000028fb: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
1000028ff: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100002904: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100002908: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000290c: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002911: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002915: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002919: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
10000291e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
100002922: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
100002927: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
10000292b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
10000292f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002934: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002938: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
10000293c: c4 e2 7d 25 64 b9 40        	vpmovsxdq	64(%rcx,%rdi,4), %ymm4
100002943: c4 e2 7d 25 6c b9 50        	vpmovsxdq	80(%rcx,%rdi,4), %ymm5
10000294a: c4 e2 7d 25 74 b9 60        	vpmovsxdq	96(%rcx,%rdi,4), %ymm6
100002951: c4 e2 7d 25 7c b9 70        	vpmovsxdq	112(%rcx,%rdi,4), %ymm7
100002958: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
10000295d: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100002962: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100002967: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
10000296b: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100002970: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100002976: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
10000297a: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
10000297f: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100002984: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100002988: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
10000298d: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100002991: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100002996: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
10000299b: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
10000299f: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
1000029a3: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
1000029a8: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
1000029ac: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
1000029b1: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
1000029b5: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000029b9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000029be: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
1000029c2: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
1000029c6: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
1000029cb: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
1000029cf: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
1000029d4: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
1000029d8: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
1000029dc: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
1000029e1: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
1000029e5: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000029e9: c4 e2 7d 25 a4 b9 80 00 00 00       	vpmovsxdq	128(%rcx,%rdi,4), %ymm4
1000029f3: c4 e2 7d 25 ac b9 90 00 00 00       	vpmovsxdq	144(%rcx,%rdi,4), %ymm5
1000029fd: c4 e2 7d 25 b4 b9 a0 00 00 00       	vpmovsxdq	160(%rcx,%rdi,4), %ymm6
100002a07: c4 e2 7d 25 bc b9 b0 00 00 00       	vpmovsxdq	176(%rcx,%rdi,4), %ymm7
100002a11: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
100002a16: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100002a1b: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100002a20: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
100002a24: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100002a29: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100002a2f: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100002a33: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100002a38: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100002a3d: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100002a41: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
100002a46: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100002a4a: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100002a4f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002a54: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100002a58: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100002a5c: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100002a61: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100002a65: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100002a6a: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100002a6e: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002a72: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002a77: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002a7b: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002a7f: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
100002a84: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100002a88: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100002a8d: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100002a91: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002a95: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002a9a: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002a9e: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100002aa2: c4 e2 7d 25 a4 b9 c0 00 00 00       	vpmovsxdq	192(%rcx,%rdi,4), %ymm4
100002aac: c4 e2 7d 25 ac b9 d0 00 00 00       	vpmovsxdq	208(%rcx,%rdi,4), %ymm5
100002ab6: c4 e2 7d 25 b4 b9 e0 00 00 00       	vpmovsxdq	224(%rcx,%rdi,4), %ymm6
100002ac0: c4 e2 7d 25 bc b9 f0 00 00 00       	vpmovsxdq	240(%rcx,%rdi,4), %ymm7
100002aca: c5 bd 73 d4 20              	vpsrlq	$32, %ymm4, %ymm8
100002acf: c4 41 7d f4 c0              	vpmuludq	%ymm8, %ymm0, %ymm8
100002ad4: c5 b5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm9
100002ad9: c5 35 f4 cc                 	vpmuludq	%ymm4, %ymm9, %ymm9
100002add: c4 41 3d d4 c1              	vpaddq	%ymm9, %ymm8, %ymm8
100002ae2: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100002ae8: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100002aec: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100002af1: c5 dd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm4
100002af6: c5 e5 f4 e4                 	vpmuludq	%ymm4, %ymm3, %ymm4
100002afa: c5 bd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm8
100002aff: c5 3d f4 c5                 	vpmuludq	%ymm5, %ymm8, %ymm8
100002b03: c4 c1 5d d4 e0              	vpaddq	%ymm8, %ymm4, %ymm4
100002b08: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002b0d: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100002b11: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100002b15: c5 dd 73 d6 20              	vpsrlq	$32, %ymm6, %ymm4
100002b1a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100002b1e: c5 d5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm5
100002b23: c5 d5 f4 ee                 	vpmuludq	%ymm6, %ymm5, %ymm5
100002b27: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002b2b: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002b30: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002b34: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002b38: c5 dd 73 d7 20              	vpsrlq	$32, %ymm7, %ymm4
100002b3d: c5 f5 f4 e4                 	vpmuludq	%ymm4, %ymm1, %ymm4
100002b41: c5 d5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm5
100002b46: c5 d5 f4 ef                 	vpmuludq	%ymm7, %ymm5, %ymm5
100002b4a: c5 dd d4 e5                 	vpaddq	%ymm5, %ymm4, %ymm4
100002b4e: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002b53: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002b57: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100002b5b: 48 83 c7 40                 	addq	$64, %rdi
100002b5f: 48 83 c6 04                 	addq	$4, %rsi
100002b63: 0f 85 27 fd ff ff           	jne	-729 <_main+0x550>
100002b69: 48 85 db                    	testq	%rbx, %rbx
100002b6c: 0f 84 c7 00 00 00           	je	199 <_main+0x8f9>
100002b72: 48 8d 34 b9                 	leaq	(%rcx,%rdi,4), %rsi
100002b76: 48 83 c6 30                 	addq	$48, %rsi
100002b7a: 48 c1 e3 06                 	shlq	$6, %rbx
100002b7e: 31 ff                       	xorl	%edi, %edi
100002b80: c4 e2 7d 25 64 3e d0        	vpmovsxdq	-48(%rsi,%rdi), %ymm4
100002b87: c4 e2 7d 25 6c 3e e0        	vpmovsxdq	-32(%rsi,%rdi), %ymm5
100002b8e: c4 e2 7d 25 74 3e f0        	vpmovsxdq	-16(%rsi,%rdi), %ymm6
100002b95: c4 e2 7d 25 3c 3e           	vpmovsxdq	(%rsi,%rdi), %ymm7
100002b9b: c5 bd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm8
100002ba0: c5 3d f4 c4                 	vpmuludq	%ymm4, %ymm8, %ymm8
100002ba4: c5 b5 73 d4 20              	vpsrlq	$32, %ymm4, %ymm9
100002ba9: c4 41 7d f4 c9              	vpmuludq	%ymm9, %ymm0, %ymm9
100002bae: c4 41 35 d4 c0              	vpaddq	%ymm8, %ymm9, %ymm8
100002bb3: c4 c1 3d 73 f0 20           	vpsllq	$32, %ymm8, %ymm8
100002bb9: c5 fd f4 c4                 	vpmuludq	%ymm4, %ymm0, %ymm0
100002bbd: c4 c1 7d d4 c0              	vpaddq	%ymm8, %ymm0, %ymm0
100002bc2: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
100002bc7: c5 dd f4 e5                 	vpmuludq	%ymm5, %ymm4, %ymm4
100002bcb: c5 bd 73 d5 20              	vpsrlq	$32, %ymm5, %ymm8
100002bd0: c4 41 65 f4 c0              	vpmuludq	%ymm8, %ymm3, %ymm8
100002bd5: c5 bd d4 e4                 	vpaddq	%ymm4, %ymm8, %ymm4
100002bd9: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002bde: c5 e5 f4 dd                 	vpmuludq	%ymm5, %ymm3, %ymm3
100002be2: c5 e5 d4 dc                 	vpaddq	%ymm4, %ymm3, %ymm3
100002be6: c5 dd 73 d2 20              	vpsrlq	$32, %ymm2, %ymm4
100002beb: c5 dd f4 e6                 	vpmuludq	%ymm6, %ymm4, %ymm4
100002bef: c5 d5 73 d6 20              	vpsrlq	$32, %ymm6, %ymm5
100002bf4: c5 ed f4 ed                 	vpmuludq	%ymm5, %ymm2, %ymm5
100002bf8: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
100002bfc: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002c01: c5 ed f4 d6                 	vpmuludq	%ymm6, %ymm2, %ymm2
100002c05: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100002c09: c5 dd 73 d1 20              	vpsrlq	$32, %ymm1, %ymm4
100002c0e: c5 dd f4 e7                 	vpmuludq	%ymm7, %ymm4, %ymm4
100002c12: c5 d5 73 d7 20              	vpsrlq	$32, %ymm7, %ymm5
100002c17: c5 f5 f4 ed                 	vpmuludq	%ymm5, %ymm1, %ymm5
100002c1b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
100002c1f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002c24: c5 f5 f4 cf                 	vpmuludq	%ymm7, %ymm1, %ymm1
100002c28: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100002c2c: 48 83 c7 40                 	addq	$64, %rdi
100002c30: 48 39 fb                    	cmpq	%rdi, %rbx
100002c33: 0f 85 47 ff ff ff           	jne	-185 <_main+0x840>
100002c39: c5 dd 73 d3 20              	vpsrlq	$32, %ymm3, %ymm4
100002c3e: c5 dd f4 e0                 	vpmuludq	%ymm0, %ymm4, %ymm4
100002c42: c5 d5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm5
100002c47: c5 e5 f4 ed                 	vpmuludq	%ymm5, %ymm3, %ymm5
100002c4b: c5 d5 d4 e4                 	vpaddq	%ymm4, %ymm5, %ymm4
100002c4f: c5 dd 73 f4 20              	vpsllq	$32, %ymm4, %ymm4
100002c54: c5 e5 f4 c0                 	vpmuludq	%ymm0, %ymm3, %ymm0
100002c58: c5 fd d4 c4                 	vpaddq	%ymm4, %ymm0, %ymm0
100002c5c: c5 e5 73 d2 20              	vpsrlq	$32, %ymm2, %ymm3
100002c61: c5 e5 f4 d8                 	vpmuludq	%ymm0, %ymm3, %ymm3
100002c65: c5 dd 73 d0 20              	vpsrlq	$32, %ymm0, %ymm4
100002c6a: c5 ed f4 e4                 	vpmuludq	%ymm4, %ymm2, %ymm4
100002c6e: c5 dd d4 db                 	vpaddq	%ymm3, %ymm4, %ymm3
100002c72: c5 e5 73 f3 20              	vpsllq	$32, %ymm3, %ymm3
100002c77: c5 ed f4 c0                 	vpmuludq	%ymm0, %ymm2, %ymm0
100002c7b: c5 fd d4 c3                 	vpaddq	%ymm3, %ymm0, %ymm0
100002c7f: c5 ed 73 d1 20              	vpsrlq	$32, %ymm1, %ymm2
100002c84: c5 ed f4 d0                 	vpmuludq	%ymm0, %ymm2, %ymm2
100002c88: c5 e5 73 d0 20              	vpsrlq	$32, %ymm0, %ymm3
100002c8d: c5 f5 f4 db                 	vpmuludq	%ymm3, %ymm1, %ymm3
100002c91: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
100002c95: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
100002c9a: c5 f5 f4 c0                 	vpmuludq	%ymm0, %ymm1, %ymm0
100002c9e: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
100002ca2: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
100002ca8: c5 ed 73 d0 20              	vpsrlq	$32, %ymm0, %ymm2
100002cad: c5 ed f4 d1                 	vpmuludq	%ymm1, %ymm2, %ymm2
100002cb1: c5 e5 73 d1 20              	vpsrlq	$32, %ymm1, %ymm3
100002cb6: c5 fd f4 db                 	vpmuludq	%ymm3, %ymm0, %ymm3
100002cba: c5 e5 d4 d2                 	vpaddq	%ymm2, %ymm3, %ymm2
100002cbe: c5 ed 73 f2 20              	vpsllq	$32, %ymm2, %ymm2
100002cc3: c5 fd f4 c1                 	vpmuludq	%ymm1, %ymm0, %ymm0
100002cc7: c5 fd d4 c2                 	vpaddq	%ymm2, %ymm0, %ymm0
100002ccb: c5 f9 70 c8 4e              	vpshufd	$78, %xmm0, %xmm1
100002cd0: c5 e9 73 d0 20              	vpsrlq	$32, %xmm0, %xmm2
100002cd5: c5 e9 f4 d1                 	vpmuludq	%xmm1, %xmm2, %xmm2
100002cd9: c5 e1 73 d8 0c              	vpsrldq	$12, %xmm0, %xmm3
100002cde: c5 f9 f4 db                 	vpmuludq	%xmm3, %xmm0, %xmm3
100002ce2: c5 e1 d4 d2                 	vpaddq	%xmm2, %xmm3, %xmm2
100002ce6: c5 e9 73 f2 20              	vpsllq	$32, %xmm2, %xmm2
100002ceb: c5 f9 f4 c1                 	vpmuludq	%xmm1, %xmm0, %xmm0
100002cef: c5 f9 d4 c2                 	vpaddq	%xmm2, %xmm0, %xmm0
100002cf3: c4 e1 f9 7e c6              	vmovq	%xmm0, %rsi
100002cf8: 48 39 c2                    	cmpq	%rax, %rdx
100002cfb: 48 8d 9c 24 10 02 00 00     	leaq	528(%rsp), %rbx
100002d03: 74 1b                       	je	27 <_main+0x9e0>
100002d05: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002d0f: 90                          	nop
100002d10: 48 63 3c 91                 	movslq	(%rcx,%rdx,4), %rdi
100002d14: 48 0f af f7                 	imulq	%rdi, %rsi
100002d18: 48 ff c2                    	incq	%rdx
100002d1b: 48 39 d0                    	cmpq	%rdx, %rax
100002d1e: 75 f0                       	jne	-16 <_main+0x9d0>
100002d20: 85 c0                       	testl	%eax, %eax
100002d22: 0f 85 9c f7 ff ff           	jne	-2148 <_main+0x184>
100002d28: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002d30: 48 8b 44 24 50              	movq	80(%rsp), %rax
100002d35: 48 85 c0                    	testq	%rax, %rax
100002d38: 74 13                       	je	19 <_main+0xa0d>
100002d3a: f0                          	lock
100002d3b: ff 48 14                    	decl	20(%rax)
100002d3e: 75 0d                       	jne	13 <_main+0xa0d>
100002d40: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
100002d45: c5 f8 77                    	vzeroupper
100002d48: e8 e9 40 00 00              	callq	16617 <dyld_stub_binder+0x100006e36>
100002d4d: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
100002d56: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100002d5a: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
100002d5f: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100002d64: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
100002d69: 7e 29                       	jle	41 <_main+0xa54>
100002d6b: 48 8b 44 24 58              	movq	88(%rsp), %rax
100002d70: 31 c9                       	xorl	%ecx, %ecx
100002d72: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002d7c: 0f 1f 40 00                 	nopl	(%rax)
100002d80: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100002d87: 48 ff c1                    	incq	%rcx
100002d8a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
100002d8f: 48 39 d1                    	cmpq	%rdx, %rcx
100002d92: 7c ec                       	jl	-20 <_main+0xa40>
100002d94: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100002d99: 4c 39 ef                    	cmpq	%r13, %rdi
100002d9c: 0f 84 8e f6 ff ff           	je	-2418 <_main+0xf0>
100002da2: c5 f8 77                    	vzeroupper
100002da5: e8 c2 40 00 00              	callq	16578 <dyld_stub_binder+0x100006e6c>
100002daa: e9 81 f6 ff ff              	jmp	-2431 <_main+0xf0>
100002daf: 8b 44 24 18                 	movl	24(%rsp), %eax
100002db3: 89 84 24 80 00 00 00        	movl	%eax, 128(%rsp)
100002dba: 8b 44 24 1c                 	movl	28(%rsp), %eax
100002dbe: 83 f8 02                    	cmpl	$2, %eax
100002dc1: 0f 8e 01 fa ff ff           	jle	-1535 <_main+0x488>
100002dc7: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
100002dcf: 48 8d 74 24 18              	leaq	24(%rsp), %rsi
100002dd4: c5 f8 77                    	vzeroupper
100002dd7: e8 66 40 00 00              	callq	16486 <dyld_stub_binder+0x100006e42>
100002ddc: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100002de0: c4 c1 eb 2a c5              	vcvtsi2sd	%r13, %xmm2, %xmm0
100002de5: c4 c1 eb 2a cf              	vcvtsi2sd	%r15, %xmm2, %xmm1
100002dea: c5 fb 10 15 8e 42 00 00     	vmovsd	17038(%rip), %xmm2
100002df2: c5 fb 5e c2                 	vdivsd	%xmm2, %xmm0, %xmm0
100002df6: c5 f3 5e ca                 	vdivsd	%xmm2, %xmm1, %xmm1
100002dfa: c5 fc 10 54 24 28           	vmovups	40(%rsp), %ymm2
100002e00: c5 fc 11 94 24 90 00 00 00  	vmovups	%ymm2, 144(%rsp)
100002e09: c5 f9 10 54 24 48           	vmovupd	72(%rsp), %xmm2
100002e0f: c5 f9 11 94 24 b0 00 00 00  	vmovupd	%xmm2, 176(%rsp)
100002e18: 85 c9                       	testl	%ecx, %ecx
100002e1a: 4d 89 f5                    	movq	%r14, %r13
100002e1d: 0f 84 53 01 00 00           	je	339 <_main+0xc36>
100002e23: 31 c0                       	xorl	%eax, %eax
100002e25: 8b 74 24 24                 	movl	36(%rsp), %esi
100002e29: 85 f6                       	testl	%esi, %esi
100002e2b: be 00 00 00 00              	movl	$0, %esi
100002e30: 75 21                       	jne	33 <_main+0xb13>
100002e32: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100002e3c: 0f 1f 40 00                 	nopl	(%rax)
100002e40: ff c0                       	incl	%eax
100002e42: 39 c8                       	cmpl	%ecx, %eax
100002e44: 0f 83 2c 01 00 00           	jae	300 <_main+0xc36>
100002e4a: 85 f6                       	testl	%esi, %esi
100002e4c: be 00 00 00 00              	movl	$0, %esi
100002e51: 74 ed                       	je	-19 <_main+0xb00>
100002e53: 48 63 c8                    	movslq	%eax, %rcx
100002e56: 31 d2                       	xorl	%edx, %edx
100002e58: c5 fb 10 25 38 42 00 00     	vmovsd	16952(%rip), %xmm4
100002e60: c5 fa 10 2d 68 42 00 00     	vmovss	17000(%rip), %xmm5
100002e68: 0f 1f 84 00 00 00 00 00     	nopl	(%rax,%rax)
100002e70: 48 8b 74 24 60              	movq	96(%rsp), %rsi
100002e75: 48 8b 3e                    	movq	(%rsi), %rdi
100002e78: 48 0f af f9                 	imulq	%rcx, %rdi
100002e7c: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100002e81: 48 63 d2                    	movslq	%edx, %rdx
100002e84: 48 8d 34 52                 	leaq	(%rdx,%rdx,2), %rsi
100002e88: 0f b6 3c 37                 	movzbl	(%rdi,%rsi), %edi
100002e8c: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100002e90: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100002e94: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100002e98: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100002ea0: 48 8b 1b                    	movq	(%rbx), %rbx
100002ea3: 48 0f af d9                 	imulq	%rcx, %rbx
100002ea7: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100002eaf: 40 88 3c 33                 	movb	%dil, (%rbx,%rsi)
100002eb3: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100002eb8: 48 8b 3f                    	movq	(%rdi), %rdi
100002ebb: 48 0f af f9                 	imulq	%rcx, %rdi
100002ebf: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100002ec4: 0f b6 7c 37 01              	movzbl	1(%rdi,%rsi), %edi
100002ec9: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100002ecd: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
100002ed5: 48 8b 3f                    	movq	(%rdi), %rdi
100002ed8: 48 0f af f9                 	imulq	%rcx, %rdi
100002edc: 48 03 bc 24 f0 00 00 00     	addq	240(%rsp), %rdi
100002ee4: 0f b6 3c 3a                 	movzbl	(%rdx,%rdi), %edi
100002ee8: c5 ca 2a df                 	vcvtsi2ss	%edi, %xmm6, %xmm3
100002eec: c5 e2 59 dd                 	vmulss	%xmm5, %xmm3, %xmm3
100002ef0: c5 e2 5a db                 	vcvtss2sd	%xmm3, %xmm3, %xmm3
100002ef4: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100002ef8: c5 eb 58 d3                 	vaddsd	%xmm3, %xmm2, %xmm2
100002efc: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100002f00: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100002f08: 48 8b 1b                    	movq	(%rbx), %rbx
100002f0b: 48 0f af d9                 	imulq	%rcx, %rbx
100002f0f: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100002f17: 40 88 7c 33 01              	movb	%dil, 1(%rbx,%rsi)
100002f1c: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100002f21: 48 8b 3f                    	movq	(%rdi), %rdi
100002f24: 48 0f af f9                 	imulq	%rcx, %rdi
100002f28: 48 03 7c 24 28              	addq	40(%rsp), %rdi
100002f2d: 0f b6 7c 37 02              	movzbl	2(%rdi,%rsi), %edi
100002f32: c5 cb 2a d7                 	vcvtsi2sd	%edi, %xmm6, %xmm2
100002f36: c5 eb 59 d4                 	vmulsd	%xmm4, %xmm2, %xmm2
100002f3a: c5 fb 2c fa                 	vcvttsd2si	%xmm2, %edi
100002f3e: 48 8b 9c 24 c8 00 00 00     	movq	200(%rsp), %rbx
100002f46: 48 8b 1b                    	movq	(%rbx), %rbx
100002f49: 48 0f af d9                 	imulq	%rcx, %rbx
100002f4d: 48 03 9c 24 90 00 00 00     	addq	144(%rsp), %rbx
100002f55: 40 88 7c 33 02              	movb	%dil, 2(%rbx,%rsi)
100002f5a: ff c2                       	incl	%edx
100002f5c: 8b 74 24 24                 	movl	36(%rsp), %esi
100002f60: 39 f2                       	cmpl	%esi, %edx
100002f62: 0f 82 08 ff ff ff           	jb	-248 <_main+0xb30>
100002f68: 8b 4c 24 20                 	movl	32(%rsp), %ecx
100002f6c: ff c0                       	incl	%eax
100002f6e: 39 c8                       	cmpl	%ecx, %eax
100002f70: 0f 82 d4 fe ff ff           	jb	-300 <_main+0xb0a>
100002f76: c5 fb 10 15 22 41 00 00     	vmovsd	16674(%rip), %xmm2
100002f7e: c5 eb 59 54 24 78           	vmulsd	120(%rsp), %xmm2, %xmm2
100002f84: c5 f3 5c c0                 	vsubsd	%xmm0, %xmm1, %xmm0
100002f88: c5 fb 58 05 18 41 00 00     	vaddsd	16664(%rip), %xmm0, %xmm0
100002f90: c5 fb 10 0d 18 41 00 00     	vmovsd	16664(%rip), %xmm1
100002f98: c5 f3 5e c0                 	vdivsd	%xmm0, %xmm1, %xmm0
100002f9c: c5 eb 58 c0                 	vaddsd	%xmm0, %xmm2, %xmm0
100002fa0: 8b 9c 24 f8 01 00 00        	movl	504(%rsp), %ebx
100002fa7: c5 fb 11 44 24 78           	vmovsd	%xmm0, 120(%rsp)
100002fad: c5 f8 77                    	vzeroupper
100002fb0: e8 47 3f 00 00              	callq	16199 <dyld_stub_binder+0x100006efc>
100002fb5: c5 fb 2c f0                 	vcvttsd2si	%xmm0, %esi
100002fb9: 4c 89 e7                    	movq	%r12, %rdi
100002fbc: e8 05 3f 00 00              	callq	16133 <dyld_stub_binder+0x100006ec6>
100002fc1: 4c 89 e7                    	movq	%r12, %rdi
100002fc4: 31 f6                       	xorl	%esi, %esi
100002fc6: 48 8d 15 38 5f 00 00        	leaq	24376(%rip), %rdx
100002fcd: e8 c4 3e 00 00              	callq	16068 <dyld_stub_binder+0x100006e96>
100002fd2: 48 8b 48 10                 	movq	16(%rax), %rcx
100002fd6: 48 89 8c 24 50 01 00 00     	movq	%rcx, 336(%rsp)
100002fde: c5 f9 10 00                 	vmovupd	(%rax), %xmm0
100002fe2: c5 f9 29 84 24 40 01 00 00  	vmovapd	%xmm0, 320(%rsp)
100002feb: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
100002fef: c5 f9 11 00                 	vmovupd	%xmm0, (%rax)
100002ff3: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
100002ffb: 48 8d bc 24 40 01 00 00     	leaq	320(%rsp), %rdi
100003003: 48 8d 35 02 5f 00 00        	leaq	24322(%rip), %rsi
10000300a: e8 7b 3e 00 00              	callq	15995 <dyld_stub_binder+0x100006e8a>
10000300f: c4 e1 cb 2a c3              	vcvtsi2sd	%rbx, %xmm6, %xmm0
100003014: c5 fb 59 44 24 78           	vmulsd	120(%rsp), %xmm0, %xmm0
10000301a: c5 fb 5e 05 96 40 00 00     	vdivsd	16534(%rip), %xmm0, %xmm0
100003022: 48 8b 48 10                 	movq	16(%rax), %rcx
100003026: 48 89 8c 24 d0 03 00 00     	movq	%rcx, 976(%rsp)
10000302e: c5 f9 10 08                 	vmovupd	(%rax), %xmm1
100003032: c5 f9 29 8c 24 c0 03 00 00  	vmovapd	%xmm1, 960(%rsp)
10000303b: c5 f1 57 c9                 	vxorpd	%xmm1, %xmm1, %xmm1
10000303f: c5 f9 11 08                 	vmovupd	%xmm1, (%rax)
100003043: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
10000304b: 48 8d bc 24 98 01 00 00     	leaq	408(%rsp), %rdi
100003053: e8 68 3e 00 00              	callq	15976 <dyld_stub_binder+0x100006ec0>
100003058: 0f b6 94 24 98 01 00 00     	movzbl	408(%rsp), %edx
100003060: f6 c2 01                    	testb	$1, %dl
100003063: 48 8d 9c 24 10 02 00 00     	leaq	528(%rsp), %rbx
10000306b: 74 12                       	je	18 <_main+0xd3f>
10000306d: 48 8b b4 24 a8 01 00 00     	movq	424(%rsp), %rsi
100003075: 48 8b 94 24 a0 01 00 00     	movq	416(%rsp), %rdx
10000307d: eb 0b                       	jmp	11 <_main+0xd4a>
10000307f: 48 d1 ea                    	shrq	%rdx
100003082: 48 8d b4 24 99 01 00 00     	leaq	409(%rsp), %rsi
10000308a: 4c 89 ef                    	movq	%r13, %rdi
10000308d: e8 fe 3d 00 00              	callq	15870 <dyld_stub_binder+0x100006e90>
100003092: 48 8b 48 10                 	movq	16(%rax), %rcx
100003096: 48 89 8c 24 70 01 00 00     	movq	%rcx, 368(%rsp)
10000309e: c5 f8 10 00                 	vmovups	(%rax), %xmm0
1000030a2: c5 f8 29 84 24 60 01 00 00  	vmovaps	%xmm0, 352(%rsp)
1000030ab: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000030af: c5 f8 11 00                 	vmovups	%xmm0, (%rax)
1000030b3: 48 c7 40 10 00 00 00 00     	movq	$0, 16(%rax)
1000030bb: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
1000030c3: 0f 85 80 01 00 00           	jne	384 <_main+0xf09>
1000030c9: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000030d1: 0f 85 8d 01 00 00           	jne	397 <_main+0xf24>
1000030d7: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
1000030df: 0f 85 9a 01 00 00           	jne	410 <_main+0xf3f>
1000030e5: 4d 89 e7                    	movq	%r12, %r15
1000030e8: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
1000030f0: 74 0d                       	je	13 <_main+0xdbf>
1000030f2: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
1000030fa: e8 d3 3d 00 00              	callq	15827 <dyld_stub_binder+0x100006ed2>
1000030ff: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
10000310b: c7 84 24 c0 03 00 00 00 00 01 03    	movl	$50397184, 960(%rsp)
100003116: 4c 8d a4 24 80 00 00 00     	leaq	128(%rsp), %r12
10000311e: 4c 89 a4 24 c8 03 00 00     	movq	%r12, 968(%rsp)
100003126: 48 b8 1e 00 00 00 1e 00 00 00       	movabsq	$128849018910, %rax
100003130: 48 89 84 24 b8 01 00 00     	movq	%rax, 440(%rsp)
100003138: c5 fc 28 05 c0 3f 00 00     	vmovaps	16320(%rip), %ymm0
100003140: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100003149: c7 44 24 08 00 00 00 00     	movl	$0, 8(%rsp)
100003151: c7 04 24 10 00 00 00        	movl	$16, (%rsp)
100003158: 4c 89 ef                    	movq	%r13, %rdi
10000315b: 48 8d b4 24 60 01 00 00     	leaq	352(%rsp), %rsi
100003163: 48 8d 94 24 b8 01 00 00     	leaq	440(%rsp), %rdx
10000316b: 31 c9                       	xorl	%ecx, %ecx
10000316d: c5 fb 10 05 4b 3f 00 00     	vmovsd	16203(%rip), %xmm0
100003175: 4c 8d 84 24 40 02 00 00     	leaq	576(%rsp), %r8
10000317d: 41 b9 02 00 00 00           	movl	$2, %r9d
100003183: c5 f8 77                    	vzeroupper
100003186: e8 cf 3c 00 00              	callq	15567 <dyld_stub_binder+0x100006e5a>
10000318b: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
10000318f: c5 f9 29 84 24 c0 03 00 00  	vmovapd	%xmm0, 960(%rsp)
100003198: 48 c7 84 24 d0 03 00 00 00 00 00 00 	movq	$0, 976(%rsp)
1000031a4: c6 84 24 c0 03 00 00 0a     	movb	$10, 960(%rsp)
1000031ac: 48 8d 84 24 c1 03 00 00     	leaq	961(%rsp), %rax
1000031b4: c6 40 04 65                 	movb	$101, 4(%rax)
1000031b8: c7 00 66 72 61 6d           	movl	$1835102822, (%rax)
1000031be: c6 84 24 c6 03 00 00 00     	movb	$0, 966(%rsp)
1000031c6: 48 c7 84 24 50 01 00 00 00 00 00 00 	movq	$0, 336(%rsp)
1000031d2: c7 84 24 40 01 00 00 00 00 01 01    	movl	$16842752, 320(%rsp)
1000031dd: 4c 89 a4 24 48 01 00 00     	movq	%r12, 328(%rsp)
1000031e5: 4c 89 ef                    	movq	%r13, %rdi
1000031e8: 48 8d b4 24 40 01 00 00     	leaq	320(%rsp), %rsi
1000031f0: e8 59 3c 00 00              	callq	15449 <dyld_stub_binder+0x100006e4e>
1000031f5: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000031fd: 4d 89 fc                    	movq	%r15, %r12
100003200: 4c 8d 6c 24 68              	leaq	104(%rsp), %r13
100003205: 0f 85 97 00 00 00           	jne	151 <_main+0xf62>
10000320b: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
100003213: 4c 8d bc 24 80 00 00 00     	leaq	128(%rsp), %r15
10000321b: 0f 85 a4 00 00 00           	jne	164 <_main+0xf85>
100003221: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
100003229: 48 85 c0                    	testq	%rax, %rax
10000322c: 0f 84 b1 00 00 00           	je	177 <_main+0xfa3>
100003232: f0                          	lock
100003233: ff 48 14                    	decl	20(%rax)
100003236: 0f 85 a7 00 00 00           	jne	167 <_main+0xfa3>
10000323c: 4c 89 ff                    	movq	%r15, %rdi
10000323f: e8 f2 3b 00 00              	callq	15346 <dyld_stub_binder+0x100006e36>
100003244: e9 9a 00 00 00              	jmp	154 <_main+0xfa3>
100003249: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100003251: e8 7c 3c 00 00              	callq	15484 <dyld_stub_binder+0x100006ed2>
100003256: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
10000325e: 0f 84 73 fe ff ff           	je	-397 <_main+0xd97>
100003264: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
10000326c: e8 61 3c 00 00              	callq	15457 <dyld_stub_binder+0x100006ed2>
100003271: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003279: 0f 84 66 fe ff ff           	je	-410 <_main+0xda5>
10000327f: 48 8b bc 24 50 01 00 00     	movq	336(%rsp), %rdi
100003287: e8 46 3c 00 00              	callq	15430 <dyld_stub_binder+0x100006ed2>
10000328c: 4d 89 e7                    	movq	%r12, %r15
10000328f: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003297: 0f 85 55 fe ff ff           	jne	-427 <_main+0xdb2>
10000329d: e9 5d fe ff ff              	jmp	-419 <_main+0xdbf>
1000032a2: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
1000032aa: e8 23 3c 00 00              	callq	15395 <dyld_stub_binder+0x100006ed2>
1000032af: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
1000032b7: 4c 8d bc 24 80 00 00 00     	leaq	128(%rsp), %r15
1000032bf: 0f 84 5c ff ff ff           	je	-164 <_main+0xee1>
1000032c5: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
1000032cd: e8 00 3c 00 00              	callq	15360 <dyld_stub_binder+0x100006ed2>
1000032d2: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000032da: 48 85 c0                    	testq	%rax, %rax
1000032dd: 0f 85 4f ff ff ff           	jne	-177 <_main+0xef2>
1000032e3: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
1000032ef: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
1000032f7: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
1000032fb: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
100003300: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
100003308: 7e 2d                       	jle	45 <_main+0xff7>
10000330a: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
100003312: 31 c9                       	xorl	%ecx, %ecx
100003314: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000331e: 66 90                       	nop
100003320: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003327: 48 ff c1                    	incq	%rcx
10000332a: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
100003332: 48 39 d1                    	cmpq	%rdx, %rcx
100003335: 7c e9                       	jl	-23 <_main+0xfe0>
100003337: 48 8b bc 24 c8 00 00 00     	movq	200(%rsp), %rdi
10000333f: 48 8d 84 24 d0 00 00 00     	leaq	208(%rsp), %rax
100003347: 48 39 c7                    	cmpq	%rax, %rdi
10000334a: 74 08                       	je	8 <_main+0x1014>
10000334c: c5 f8 77                    	vzeroupper
10000334f: e8 18 3b 00 00              	callq	15128 <dyld_stub_binder+0x100006e6c>
100003354: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
10000335c: 48 85 c0                    	testq	%rax, %rax
10000335f: 74 16                       	je	22 <_main+0x1037>
100003361: f0                          	lock
100003362: ff 48 14                    	decl	20(%rax)
100003365: 75 10                       	jne	16 <_main+0x1037>
100003367: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
10000336f: c5 f8 77                    	vzeroupper
100003372: e8 bf 3a 00 00              	callq	15039 <dyld_stub_binder+0x100006e36>
100003377: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100003383: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
10000338b: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
10000338f: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
100003393: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
10000339b: 7e 2a                       	jle	42 <_main+0x1087>
10000339d: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
1000033a5: 31 c9                       	xorl	%ecx, %ecx
1000033a7: 66 0f 1f 84 00 00 00 00 00  	nopw	(%rax,%rax)
1000033b0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000033b7: 48 ff c1                    	incq	%rcx
1000033ba: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
1000033c2: 48 39 d1                    	cmpq	%rdx, %rcx
1000033c5: 7c e9                       	jl	-23 <_main+0x1070>
1000033c7: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
1000033cf: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
1000033d7: 48 39 c7                    	cmpq	%rax, %rdi
1000033da: 74 08                       	je	8 <_main+0x10a4>
1000033dc: c5 f8 77                    	vzeroupper
1000033df: e8 88 3a 00 00              	callq	14984 <dyld_stub_binder+0x100006e6c>
1000033e4: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
1000033ec: c5 f8 77                    	vzeroupper
1000033ef: e8 0c 05 00 00              	callq	1292 <_main+0x15c0>
1000033f4: 45 31 ff                    	xorl	%r15d, %r15d
1000033f7: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000033fc: 48 85 c0                    	testq	%rax, %rax
1000033ff: 0f 85 35 f9 ff ff           	jne	-1739 <_main+0x9fa>
100003405: e9 43 f9 ff ff              	jmp	-1725 <_main+0xa0d>
10000340a: 48 8b 3d 2f 5c 00 00        	movq	23599(%rip), %rdi
100003411: 48 8d 35 08 5b 00 00        	leaq	23304(%rip), %rsi
100003418: ba 0d 00 00 00              	movl	$13, %edx
10000341d: c5 f8 77                    	vzeroupper
100003420: e8 0b 07 00 00              	callq	1803 <_main+0x17f0>
100003425: 48 8d bc 24 10 02 00 00     	leaq	528(%rsp), %rdi
10000342d: e8 f8 39 00 00              	callq	14840 <dyld_stub_binder+0x100006e2a>
100003432: 48 8b 05 17 5c 00 00        	movq	23575(%rip), %rax
100003439: 48 83 c0 10                 	addq	$16, %rax
10000343d: 48 89 84 24 d8 01 00 00     	movq	%rax, 472(%rsp)
100003445: 48 8b bc 24 00 02 00 00     	movq	512(%rsp), %rdi
10000344d: 48 85 ff                    	testq	%rdi, %rdi
100003450: 74 05                       	je	5 <_main+0x1117>
100003452: e8 7b 3a 00 00              	callq	14971 <dyld_stub_binder+0x100006ed2>
100003457: 48 8b bc 24 08 02 00 00     	movq	520(%rsp), %rdi
10000345f: 48 85 ff                    	testq	%rdi, %rdi
100003462: 74 05                       	je	5 <_main+0x1129>
100003464: e8 69 3a 00 00              	callq	14953 <dyld_stub_binder+0x100006ed2>
100003469: 48 8b 05 f0 5b 00 00        	movq	23536(%rip), %rax
100003470: 48 8b 00                    	movq	(%rax), %rax
100003473: 48 3b 84 24 e0 03 00 00     	cmpq	992(%rsp), %rax
10000347b: 75 11                       	jne	17 <_main+0x114e>
10000347d: 31 c0                       	xorl	%eax, %eax
10000347f: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100003483: 5b                          	popq	%rbx
100003484: 41 5c                       	popq	%r12
100003486: 41 5d                       	popq	%r13
100003488: 41 5e                       	popq	%r14
10000348a: 41 5f                       	popq	%r15
10000348c: 5d                          	popq	%rbp
10000348d: c3                          	retq
10000348e: e8 5d 3a 00 00              	callq	14941 <dyld_stub_binder+0x100006ef0>
100003493: e9 ed 03 00 00              	jmp	1005 <_main+0x1545>
100003498: 48 89 c3                    	movq	%rax, %rbx
10000349b: f6 84 24 60 02 00 00 01     	testb	$1, 608(%rsp)
1000034a3: 0f 84 ef 03 00 00           	je	1007 <_main+0x1558>
1000034a9: 48 8b bc 24 70 02 00 00     	movq	624(%rsp), %rdi
1000034b1: e8 1c 3a 00 00              	callq	14876 <dyld_stub_binder+0x100006ed2>
1000034b6: e9 dd 03 00 00              	jmp	989 <_main+0x1558>
1000034bb: 48 89 c3                    	movq	%rax, %rbx
1000034be: e9 d5 03 00 00              	jmp	981 <_main+0x1558>
1000034c3: 48 89 c7                    	movq	%rax, %rdi
1000034c6: e8 25 04 00 00              	callq	1061 <_main+0x15b0>
1000034cb: 48 89 c7                    	movq	%rax, %rdi
1000034ce: e8 1d 04 00 00              	callq	1053 <_main+0x15b0>
1000034d3: 48 89 c7                    	movq	%rax, %rdi
1000034d6: e8 15 04 00 00              	callq	1045 <_main+0x15b0>
1000034db: 48 89 c3                    	movq	%rax, %rbx
1000034de: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000034e6: 48 85 c0                    	testq	%rax, %rax
1000034e9: 0f 85 c8 01 00 00           	jne	456 <_main+0x1377>
1000034ef: e9 d6 01 00 00              	jmp	470 <_main+0x138a>
1000034f4: 48 89 c3                    	movq	%rax, %rbx
1000034f7: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
1000034ff: 48 85 c0                    	testq	%rax, %rax
100003502: 74 13                       	je	19 <_main+0x11d7>
100003504: f0                          	lock
100003505: ff 48 14                    	decl	20(%rax)
100003508: 75 0d                       	jne	13 <_main+0x11d7>
10000350a: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003512: e8 1f 39 00 00              	callq	14623 <dyld_stub_binder+0x100006e36>
100003517: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100003523: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003527: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
10000352f: c5 fe 7f 00                 	vmovdqu	%ymm0, (%rax)
100003533: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
10000353b: 7e 21                       	jle	33 <_main+0x121e>
10000353d: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100003545: 31 c9                       	xorl	%ecx, %ecx
100003547: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
10000354e: 48 ff c1                    	incq	%rcx
100003551: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
100003559: 48 39 d1                    	cmpq	%rdx, %rcx
10000355c: 7c e9                       	jl	-23 <_main+0x1207>
10000355e: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
100003566: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
10000356e: 48 39 c7                    	cmpq	%rax, %rdi
100003571: 0f 84 96 02 00 00           	je	662 <_main+0x14cd>
100003577: c5 f8 77                    	vzeroupper
10000357a: e8 ed 38 00 00              	callq	14573 <dyld_stub_binder+0x100006e6c>
10000357f: e9 89 02 00 00              	jmp	649 <_main+0x14cd>
100003584: 48 89 c7                    	movq	%rax, %rdi
100003587: e8 64 03 00 00              	callq	868 <_main+0x15b0>
10000358c: 48 89 c3                    	movq	%rax, %rbx
10000358f: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003594: 48 85 c0                    	testq	%rax, %rax
100003597: 0f 85 7a 02 00 00           	jne	634 <_main+0x14d7>
10000359d: e9 88 02 00 00              	jmp	648 <_main+0x14ea>
1000035a2: 48 89 c3                    	movq	%rax, %rbx
1000035a5: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000035ad: 74 1f                       	je	31 <_main+0x128e>
1000035af: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
1000035b7: e8 16 39 00 00              	callq	14614 <dyld_stub_binder+0x100006ed2>
1000035bc: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
1000035c4: 75 16                       	jne	22 <_main+0x129c>
1000035c6: e9 df 00 00 00              	jmp	223 <_main+0x136a>
1000035cb: 48 89 c3                    	movq	%rax, %rbx
1000035ce: f6 84 24 60 01 00 00 01     	testb	$1, 352(%rsp)
1000035d6: 0f 84 ce 00 00 00           	je	206 <_main+0x136a>
1000035dc: 48 8b bc 24 70 01 00 00     	movq	368(%rsp), %rdi
1000035e4: e9 aa 00 00 00              	jmp	170 <_main+0x1353>
1000035e9: 48 89 c3                    	movq	%rax, %rbx
1000035ec: f6 84 24 98 01 00 00 01     	testb	$1, 408(%rsp)
1000035f4: 75 23                       	jne	35 <_main+0x12d9>
1000035f6: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
1000035fe: 75 3f                       	jne	63 <_main+0x12ff>
100003600: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003608: 75 5b                       	jne	91 <_main+0x1325>
10000360a: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003612: 75 77                       	jne	119 <_main+0x134b>
100003614: e9 91 00 00 00              	jmp	145 <_main+0x136a>
100003619: 48 8b bc 24 a8 01 00 00     	movq	424(%rsp), %rdi
100003621: e8 ac 38 00 00              	callq	14508 <dyld_stub_binder+0x100006ed2>
100003626: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
10000362e: 74 d0                       	je	-48 <_main+0x12c0>
100003630: eb 0d                       	jmp	13 <_main+0x12ff>
100003632: 48 89 c3                    	movq	%rax, %rbx
100003635: f6 84 24 c0 03 00 00 01     	testb	$1, 960(%rsp)
10000363d: 74 c1                       	je	-63 <_main+0x12c0>
10000363f: 48 8b bc 24 d0 03 00 00     	movq	976(%rsp), %rdi
100003647: e8 86 38 00 00              	callq	14470 <dyld_stub_binder+0x100006ed2>
10000364c: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003654: 74 b4                       	je	-76 <_main+0x12ca>
100003656: eb 0d                       	jmp	13 <_main+0x1325>
100003658: 48 89 c3                    	movq	%rax, %rbx
10000365b: f6 84 24 40 01 00 00 01     	testb	$1, 320(%rsp)
100003663: 74 a5                       	je	-91 <_main+0x12ca>
100003665: 48 8b bc 24 50 01 00 00     	movq	336(%rsp), %rdi
10000366d: e8 60 38 00 00              	callq	14432 <dyld_stub_binder+0x100006ed2>
100003672: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
10000367a: 75 0f                       	jne	15 <_main+0x134b>
10000367c: eb 2c                       	jmp	44 <_main+0x136a>
10000367e: 48 89 c3                    	movq	%rax, %rbx
100003681: f6 84 24 c0 01 00 00 01     	testb	$1, 448(%rsp)
100003689: 74 1f                       	je	31 <_main+0x136a>
10000368b: 48 8b bc 24 d0 01 00 00     	movq	464(%rsp), %rdi
100003693: e8 3a 38 00 00              	callq	14394 <dyld_stub_binder+0x100006ed2>
100003698: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000036a0: 48 85 c0                    	testq	%rax, %rax
1000036a3: 75 12                       	jne	18 <_main+0x1377>
1000036a5: eb 23                       	jmp	35 <_main+0x138a>
1000036a7: 48 89 c3                    	movq	%rax, %rbx
1000036aa: 48 8b 84 24 b8 00 00 00     	movq	184(%rsp), %rax
1000036b2: 48 85 c0                    	testq	%rax, %rax
1000036b5: 74 13                       	je	19 <_main+0x138a>
1000036b7: f0                          	lock
1000036b8: ff 48 14                    	decl	20(%rax)
1000036bb: 75 0d                       	jne	13 <_main+0x138a>
1000036bd: 48 8d bc 24 80 00 00 00     	leaq	128(%rsp), %rdi
1000036c5: e8 6c 37 00 00              	callq	14188 <dyld_stub_binder+0x100006e36>
1000036ca: 48 c7 84 24 b8 00 00 00 00 00 00 00 	movq	$0, 184(%rsp)
1000036d6: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
1000036da: 48 8d 84 24 84 00 00 00     	leaq	132(%rsp), %rax
1000036e2: c5 fd 11 40 0c              	vmovupd	%ymm0, 12(%rax)
1000036e7: 83 bc 24 84 00 00 00 00     	cmpl	$0, 132(%rsp)
1000036ef: 7e 21                       	jle	33 <_main+0x13d2>
1000036f1: 48 8b 84 24 c0 00 00 00     	movq	192(%rsp), %rax
1000036f9: 31 c9                       	xorl	%ecx, %ecx
1000036fb: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003702: 48 ff c1                    	incq	%rcx
100003705: 48 63 94 24 84 00 00 00     	movslq	132(%rsp), %rdx
10000370d: 48 39 d1                    	cmpq	%rdx, %rcx
100003710: 7c e9                       	jl	-23 <_main+0x13bb>
100003712: 48 8b bc 24 c8 00 00 00     	movq	200(%rsp), %rdi
10000371a: 48 8d 84 24 d0 00 00 00     	leaq	208(%rsp), %rax
100003722: 48 39 c7                    	cmpq	%rax, %rdi
100003725: 74 21                       	je	33 <_main+0x1408>
100003727: c5 f8 77                    	vzeroupper
10000372a: e8 3d 37 00 00              	callq	14141 <dyld_stub_binder+0x100006e6c>
10000372f: eb 17                       	jmp	23 <_main+0x1408>
100003731: 48 89 c7                    	movq	%rax, %rdi
100003734: e8 b7 01 00 00              	callq	439 <_main+0x15b0>
100003739: eb 0a                       	jmp	10 <_main+0x1405>
10000373b: eb 08                       	jmp	8 <_main+0x1405>
10000373d: 48 89 c3                    	movq	%rax, %rbx
100003740: e9 8a 00 00 00              	jmp	138 <_main+0x148f>
100003745: 48 89 c3                    	movq	%rax, %rbx
100003748: 48 8b 84 24 18 01 00 00     	movq	280(%rsp), %rax
100003750: 48 85 c0                    	testq	%rax, %rax
100003753: 74 16                       	je	22 <_main+0x142b>
100003755: f0                          	lock
100003756: ff 48 14                    	decl	20(%rax)
100003759: 75 10                       	jne	16 <_main+0x142b>
10000375b: 48 8d bc 24 e0 00 00 00     	leaq	224(%rsp), %rdi
100003763: c5 f8 77                    	vzeroupper
100003766: e8 cb 36 00 00              	callq	14027 <dyld_stub_binder+0x100006e36>
10000376b: 48 c7 84 24 18 01 00 00 00 00 00 00 	movq	$0, 280(%rsp)
100003777: c5 f9 57 c0                 	vxorpd	%xmm0, %xmm0, %xmm0
10000377b: 48 8d 84 24 f0 00 00 00     	leaq	240(%rsp), %rax
100003783: c5 fd 11 00                 	vmovupd	%ymm0, (%rax)
100003787: 83 bc 24 e4 00 00 00 00     	cmpl	$0, 228(%rsp)
10000378f: 7e 21                       	jle	33 <_main+0x1472>
100003791: 48 8b 84 24 20 01 00 00     	movq	288(%rsp), %rax
100003799: 31 c9                       	xorl	%ecx, %ecx
10000379b: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000037a2: 48 ff c1                    	incq	%rcx
1000037a5: 48 63 94 24 e4 00 00 00     	movslq	228(%rsp), %rdx
1000037ad: 48 39 d1                    	cmpq	%rdx, %rcx
1000037b0: 7c e9                       	jl	-23 <_main+0x145b>
1000037b2: 48 8b bc 24 28 01 00 00     	movq	296(%rsp), %rdi
1000037ba: 48 8d 84 24 30 01 00 00     	leaq	304(%rsp), %rax
1000037c2: 48 39 c7                    	cmpq	%rax, %rdi
1000037c5: 74 08                       	je	8 <_main+0x148f>
1000037c7: c5 f8 77                    	vzeroupper
1000037ca: e8 9d 36 00 00              	callq	13981 <dyld_stub_binder+0x100006e6c>
1000037cf: 48 8d bc 24 60 02 00 00     	leaq	608(%rsp), %rdi
1000037d7: c5 f8 77                    	vzeroupper
1000037da: e8 21 01 00 00              	callq	289 <_main+0x15c0>
1000037df: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000037e4: 48 85 c0                    	testq	%rax, %rax
1000037e7: 75 2e                       	jne	46 <_main+0x14d7>
1000037e9: eb 3f                       	jmp	63 <_main+0x14ea>
1000037eb: 48 89 c7                    	movq	%rax, %rdi
1000037ee: e8 fd 00 00 00              	callq	253 <_main+0x15b0>
1000037f3: 48 89 c3                    	movq	%rax, %rbx
1000037f6: 48 8b 44 24 50              	movq	80(%rsp), %rax
1000037fb: 48 85 c0                    	testq	%rax, %rax
1000037fe: 75 17                       	jne	23 <_main+0x14d7>
100003800: eb 28                       	jmp	40 <_main+0x14ea>
100003802: 48 89 c7                    	movq	%rax, %rdi
100003805: e8 e6 00 00 00              	callq	230 <_main+0x15b0>
10000380a: 48 89 c3                    	movq	%rax, %rbx
10000380d: 48 8b 44 24 50              	movq	80(%rsp), %rax
100003812: 48 85 c0                    	testq	%rax, %rax
100003815: 74 13                       	je	19 <_main+0x14ea>
100003817: f0                          	lock
100003818: ff 48 14                    	decl	20(%rax)
10000381b: 75 0d                       	jne	13 <_main+0x14ea>
10000381d: 48 8d 7c 24 18              	leaq	24(%rsp), %rdi
100003822: c5 f8 77                    	vzeroupper
100003825: e8 0c 36 00 00              	callq	13836 <dyld_stub_binder+0x100006e36>
10000382a: 48 c7 44 24 50 00 00 00 00  	movq	$0, 80(%rsp)
100003833: c5 f9 ef c0                 	vpxor	%xmm0, %xmm0, %xmm0
100003837: 48 8d 44 24 1c              	leaq	28(%rsp), %rax
10000383c: c5 fe 7f 40 0c              	vmovdqu	%ymm0, 12(%rax)
100003841: 83 7c 24 1c 00              	cmpl	$0, 28(%rsp)
100003846: 7e 1c                       	jle	28 <_main+0x1524>
100003848: 48 8b 44 24 58              	movq	88(%rsp), %rax
10000384d: 31 c9                       	xorl	%ecx, %ecx
10000384f: 90                          	nop
100003850: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003857: 48 ff c1                    	incq	%rcx
10000385a: 48 63 54 24 1c              	movslq	28(%rsp), %rdx
10000385f: 48 39 d1                    	cmpq	%rdx, %rcx
100003862: 7c ec                       	jl	-20 <_main+0x1510>
100003864: 48 8b 7c 24 60              	movq	96(%rsp), %rdi
100003869: 48 8d 44 24 68              	leaq	104(%rsp), %rax
10000386e: 48 39 c7                    	cmpq	%rax, %rdi
100003871: 74 15                       	je	21 <_main+0x1548>
100003873: c5 f8 77                    	vzeroupper
100003876: e8 f1 35 00 00              	callq	13809 <dyld_stub_binder+0x100006e6c>
10000387b: eb 0b                       	jmp	11 <_main+0x1548>
10000387d: 48 89 c7                    	movq	%rax, %rdi
100003880: e8 6b 00 00 00              	callq	107 <_main+0x15b0>
100003885: 48 89 c3                    	movq	%rax, %rbx
100003888: 48 8d bc 24 10 02 00 00     	leaq	528(%rsp), %rdi
100003890: c5 f8 77                    	vzeroupper
100003893: e8 92 35 00 00              	callq	13714 <dyld_stub_binder+0x100006e2a>
100003898: 48 8b 05 b1 57 00 00        	movq	22449(%rip), %rax
10000389f: 48 83 c0 10                 	addq	$16, %rax
1000038a3: 48 89 84 24 d8 01 00 00     	movq	%rax, 472(%rsp)
1000038ab: 48 8b bc 24 00 02 00 00     	movq	512(%rsp), %rdi
1000038b3: 48 85 ff                    	testq	%rdi, %rdi
1000038b6: 75 17                       	jne	23 <_main+0x158f>
1000038b8: 48 8b bc 24 08 02 00 00     	movq	520(%rsp), %rdi
1000038c0: 48 85 ff                    	testq	%rdi, %rdi
1000038c3: 75 1c                       	jne	28 <_main+0x15a1>
1000038c5: 48 89 df                    	movq	%rbx, %rdi
1000038c8: e8 4b 35 00 00              	callq	13643 <dyld_stub_binder+0x100006e18>
1000038cd: 0f 0b                       	ud2
1000038cf: e8 fe 35 00 00              	callq	13822 <dyld_stub_binder+0x100006ed2>
1000038d4: 48 8b bc 24 08 02 00 00     	movq	520(%rsp), %rdi
1000038dc: 48 85 ff                    	testq	%rdi, %rdi
1000038df: 74 e4                       	je	-28 <_main+0x1585>
1000038e1: e8 ec 35 00 00              	callq	13804 <dyld_stub_binder+0x100006ed2>
1000038e6: 48 89 df                    	movq	%rbx, %rdi
1000038e9: e8 2a 35 00 00              	callq	13610 <dyld_stub_binder+0x100006e18>
1000038ee: 0f 0b                       	ud2
1000038f0: 50                          	pushq	%rax
1000038f1: e8 ee 35 00 00              	callq	13806 <dyld_stub_binder+0x100006ee4>
1000038f6: e8 d1 35 00 00              	callq	13777 <dyld_stub_binder+0x100006ecc>
1000038fb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003900: 55                          	pushq	%rbp
100003901: 48 89 e5                    	movq	%rsp, %rbp
100003904: 53                          	pushq	%rbx
100003905: 50                          	pushq	%rax
100003906: 48 89 fb                    	movq	%rdi, %rbx
100003909: 48 8b 87 08 01 00 00        	movq	264(%rdi), %rax
100003910: 48 85 c0                    	testq	%rax, %rax
100003913: 74 12                       	je	18 <_main+0x15e7>
100003915: f0                          	lock
100003916: ff 48 14                    	decl	20(%rax)
100003919: 75 0c                       	jne	12 <_main+0x15e7>
10000391b: 48 8d bb d0 00 00 00        	leaq	208(%rbx), %rdi
100003922: e8 0f 35 00 00              	callq	13583 <dyld_stub_binder+0x100006e36>
100003927: 48 c7 83 08 01 00 00 00 00 00 00    	movq	$0, 264(%rbx)
100003932: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003936: c5 fc 11 83 e0 00 00 00     	vmovups	%ymm0, 224(%rbx)
10000393e: 83 bb d4 00 00 00 00        	cmpl	$0, 212(%rbx)
100003945: 7e 1f                       	jle	31 <_main+0x1626>
100003947: 48 8b 83 10 01 00 00        	movq	272(%rbx), %rax
10000394e: 31 c9                       	xorl	%ecx, %ecx
100003950: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003957: 48 ff c1                    	incq	%rcx
10000395a: 48 63 93 d4 00 00 00        	movslq	212(%rbx), %rdx
100003961: 48 39 d1                    	cmpq	%rdx, %rcx
100003964: 7c ea                       	jl	-22 <_main+0x1610>
100003966: 48 8b bb 18 01 00 00        	movq	280(%rbx), %rdi
10000396d: 48 8d 83 20 01 00 00        	leaq	288(%rbx), %rax
100003974: 48 39 c7                    	cmpq	%rax, %rdi
100003977: 74 08                       	je	8 <_main+0x1641>
100003979: c5 f8 77                    	vzeroupper
10000397c: e8 eb 34 00 00              	callq	13547 <dyld_stub_binder+0x100006e6c>
100003981: 48 8b 83 a8 00 00 00        	movq	168(%rbx), %rax
100003988: 48 85 c0                    	testq	%rax, %rax
10000398b: 74 12                       	je	18 <_main+0x165f>
10000398d: f0                          	lock
10000398e: ff 48 14                    	decl	20(%rax)
100003991: 75 0c                       	jne	12 <_main+0x165f>
100003993: 48 8d 7b 70                 	leaq	112(%rbx), %rdi
100003997: c5 f8 77                    	vzeroupper
10000399a: e8 97 34 00 00              	callq	13463 <dyld_stub_binder+0x100006e36>
10000399f: 48 c7 83 a8 00 00 00 00 00 00 00    	movq	$0, 168(%rbx)
1000039aa: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
1000039ae: c5 fc 11 83 80 00 00 00     	vmovups	%ymm0, 128(%rbx)
1000039b6: 83 7b 74 00                 	cmpl	$0, 116(%rbx)
1000039ba: 7e 27                       	jle	39 <_main+0x16a3>
1000039bc: 48 8b 83 b0 00 00 00        	movq	176(%rbx), %rax
1000039c3: 31 c9                       	xorl	%ecx, %ecx
1000039c5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000039cf: 90                          	nop
1000039d0: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
1000039d7: 48 ff c1                    	incq	%rcx
1000039da: 48 63 53 74                 	movslq	116(%rbx), %rdx
1000039de: 48 39 d1                    	cmpq	%rdx, %rcx
1000039e1: 7c ed                       	jl	-19 <_main+0x1690>
1000039e3: 48 8b bb b8 00 00 00        	movq	184(%rbx), %rdi
1000039ea: 48 8d 83 c0 00 00 00        	leaq	192(%rbx), %rax
1000039f1: 48 39 c7                    	cmpq	%rax, %rdi
1000039f4: 74 08                       	je	8 <_main+0x16be>
1000039f6: c5 f8 77                    	vzeroupper
1000039f9: e8 6e 34 00 00              	callq	13422 <dyld_stub_binder+0x100006e6c>
1000039fe: 48 8b 43 48                 	movq	72(%rbx), %rax
100003a02: 48 85 c0                    	testq	%rax, %rax
100003a05: 74 12                       	je	18 <_main+0x16d9>
100003a07: f0                          	lock
100003a08: ff 48 14                    	decl	20(%rax)
100003a0b: 75 0c                       	jne	12 <_main+0x16d9>
100003a0d: 48 8d 7b 10                 	leaq	16(%rbx), %rdi
100003a11: c5 f8 77                    	vzeroupper
100003a14: e8 1d 34 00 00              	callq	13341 <dyld_stub_binder+0x100006e36>
100003a19: 48 c7 43 48 00 00 00 00     	movq	$0, 72(%rbx)
100003a21: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003a25: c5 fc 11 43 20              	vmovups	%ymm0, 32(%rbx)
100003a2a: 83 7b 14 00                 	cmpl	$0, 20(%rbx)
100003a2e: 7e 23                       	jle	35 <_main+0x1713>
100003a30: 48 8b 43 50                 	movq	80(%rbx), %rax
100003a34: 31 c9                       	xorl	%ecx, %ecx
100003a36: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003a40: c7 04 88 00 00 00 00        	movl	$0, (%rax,%rcx,4)
100003a47: 48 ff c1                    	incq	%rcx
100003a4a: 48 63 53 14                 	movslq	20(%rbx), %rdx
100003a4e: 48 39 d1                    	cmpq	%rdx, %rcx
100003a51: 7c ed                       	jl	-19 <_main+0x1700>
100003a53: 48 8b 7b 58                 	movq	88(%rbx), %rdi
100003a57: 48 83 c3 60                 	addq	$96, %rbx
100003a5b: 48 39 df                    	cmpq	%rbx, %rdi
100003a5e: 74 08                       	je	8 <_main+0x1728>
100003a60: c5 f8 77                    	vzeroupper
100003a63: e8 04 34 00 00              	callq	13316 <dyld_stub_binder+0x100006e6c>
100003a68: 48 83 c4 08                 	addq	$8, %rsp
100003a6c: 5b                          	popq	%rbx
100003a6d: 5d                          	popq	%rbp
100003a6e: c5 f8 77                    	vzeroupper
100003a71: c3                          	retq
100003a72: 48 89 c7                    	movq	%rax, %rdi
100003a75: e8 76 fe ff ff              	callq	-394 <_main+0x15b0>
100003a7a: 48 89 c7                    	movq	%rax, %rdi
100003a7d: e8 6e fe ff ff              	callq	-402 <_main+0x15b0>
100003a82: 48 89 c7                    	movq	%rax, %rdi
100003a85: e8 66 fe ff ff              	callq	-410 <_main+0x15b0>
100003a8a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)
100003a90: 55                          	pushq	%rbp
100003a91: 48 89 e5                    	movq	%rsp, %rbp
100003a94: 53                          	pushq	%rbx
100003a95: 50                          	pushq	%rax
100003a96: 48 89 fb                    	movq	%rdi, %rbx
100003a99: 48 8b 05 b0 55 00 00        	movq	21936(%rip), %rax
100003aa0: 48 83 c0 10                 	addq	$16, %rax
100003aa4: 48 89 07                    	movq	%rax, (%rdi)
100003aa7: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100003aab: 48 85 ff                    	testq	%rdi, %rdi
100003aae: 74 05                       	je	5 <_main+0x1775>
100003ab0: e8 1d 34 00 00              	callq	13341 <dyld_stub_binder+0x100006ed2>
100003ab5: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100003ab9: 48 83 c4 08                 	addq	$8, %rsp
100003abd: 48 85 ff                    	testq	%rdi, %rdi
100003ac0: 74 07                       	je	7 <_main+0x1789>
100003ac2: 5b                          	popq	%rbx
100003ac3: 5d                          	popq	%rbp
100003ac4: e9 09 34 00 00              	jmp	13321 <dyld_stub_binder+0x100006ed2>
100003ac9: 5b                          	popq	%rbx
100003aca: 5d                          	popq	%rbp
100003acb: c3                          	retq
100003acc: 0f 1f 40 00                 	nopl	(%rax)
100003ad0: 55                          	pushq	%rbp
100003ad1: 48 89 e5                    	movq	%rsp, %rbp
100003ad4: 53                          	pushq	%rbx
100003ad5: 50                          	pushq	%rax
100003ad6: 48 89 fb                    	movq	%rdi, %rbx
100003ad9: 48 8b 05 70 55 00 00        	movq	21872(%rip), %rax
100003ae0: 48 83 c0 10                 	addq	$16, %rax
100003ae4: 48 89 07                    	movq	%rax, (%rdi)
100003ae7: 48 8b 7f 28                 	movq	40(%rdi), %rdi
100003aeb: 48 85 ff                    	testq	%rdi, %rdi
100003aee: 74 05                       	je	5 <_main+0x17b5>
100003af0: e8 dd 33 00 00              	callq	13277 <dyld_stub_binder+0x100006ed2>
100003af5: 48 8b 7b 30                 	movq	48(%rbx), %rdi
100003af9: 48 85 ff                    	testq	%rdi, %rdi
100003afc: 74 05                       	je	5 <_main+0x17c3>
100003afe: e8 cf 33 00 00              	callq	13263 <dyld_stub_binder+0x100006ed2>
100003b03: 48 89 df                    	movq	%rbx, %rdi
100003b06: 48 83 c4 08                 	addq	$8, %rsp
100003b0a: 5b                          	popq	%rbx
100003b0b: 5d                          	popq	%rbp
100003b0c: e9 c1 33 00 00              	jmp	13249 <dyld_stub_binder+0x100006ed2>
100003b11: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003b1b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003b20: 55                          	pushq	%rbp
100003b21: 48 89 e5                    	movq	%rsp, %rbp
100003b24: 5d                          	popq	%rbp
100003b25: c3                          	retq
100003b26: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003b30: 55                          	pushq	%rbp
100003b31: 48 89 e5                    	movq	%rsp, %rbp
100003b34: 41 57                       	pushq	%r15
100003b36: 41 56                       	pushq	%r14
100003b38: 41 55                       	pushq	%r13
100003b3a: 41 54                       	pushq	%r12
100003b3c: 53                          	pushq	%rbx
100003b3d: 48 83 ec 28                 	subq	$40, %rsp
100003b41: 49 89 d6                    	movq	%rdx, %r14
100003b44: 49 89 f7                    	movq	%rsi, %r15
100003b47: 48 89 fb                    	movq	%rdi, %rbx
100003b4a: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100003b4e: 48 89 de                    	movq	%rbx, %rsi
100003b51: e8 46 33 00 00              	callq	13126 <dyld_stub_binder+0x100006e9c>
100003b56: 80 7d b0 00                 	cmpb	$0, -80(%rbp)
100003b5a: 0f 84 ae 00 00 00           	je	174 <_main+0x18ce>
100003b60: 48 8b 03                    	movq	(%rbx), %rax
100003b63: 48 8b 40 e8                 	movq	-24(%rax), %rax
100003b67: 4c 8d 24 03                 	leaq	(%rbx,%rax), %r12
100003b6b: 48 8b 7c 03 28              	movq	40(%rbx,%rax), %rdi
100003b70: 44 8b 6c 03 08              	movl	8(%rbx,%rax), %r13d
100003b75: 8b 84 03 90 00 00 00        	movl	144(%rbx,%rax), %eax
100003b7c: 83 f8 ff                    	cmpl	$-1, %eax
100003b7f: 75 4a                       	jne	74 <_main+0x188b>
100003b81: 48 89 7d c0                 	movq	%rdi, -64(%rbp)
100003b85: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100003b89: 4c 89 e6                    	movq	%r12, %rsi
100003b8c: e8 f3 32 00 00              	callq	13043 <dyld_stub_binder+0x100006e84>
100003b91: 48 8b 35 b0 54 00 00        	movq	21680(%rip), %rsi
100003b98: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100003b9c: e8 dd 32 00 00              	callq	13021 <dyld_stub_binder+0x100006e7e>
100003ba1: 48 8b 08                    	movq	(%rax), %rcx
100003ba4: 48 89 c7                    	movq	%rax, %rdi
100003ba7: be 20 00 00 00              	movl	$32, %esi
100003bac: ff 51 38                    	callq	*56(%rcx)
100003baf: 88 45 d7                    	movb	%al, -41(%rbp)
100003bb2: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100003bb6: e8 f3 32 00 00              	callq	13043 <dyld_stub_binder+0x100006eae>
100003bbb: 0f be 45 d7                 	movsbl	-41(%rbp), %eax
100003bbf: 41 89 84 24 90 00 00 00     	movl	%eax, 144(%r12)
100003bc7: 48 8b 7d c0                 	movq	-64(%rbp), %rdi
100003bcb: 4d 01 fe                    	addq	%r15, %r14
100003bce: 41 81 e5 b0 00 00 00        	andl	$176, %r13d
100003bd5: 41 83 fd 20                 	cmpl	$32, %r13d
100003bd9: 4c 89 fa                    	movq	%r15, %rdx
100003bdc: 49 0f 44 d6                 	cmoveq	%r14, %rdx
100003be0: 44 0f be c8                 	movsbl	%al, %r9d
100003be4: 4c 89 fe                    	movq	%r15, %rsi
100003be7: 4c 89 f1                    	movq	%r14, %rcx
100003bea: 4d 89 e0                    	movq	%r12, %r8
100003bed: e8 9e 00 00 00              	callq	158 <_main+0x1950>
100003bf2: 48 85 c0                    	testq	%rax, %rax
100003bf5: 75 17                       	jne	23 <_main+0x18ce>
100003bf7: 48 8b 03                    	movq	(%rbx), %rax
100003bfa: 48 8b 40 e8                 	movq	-24(%rax), %rax
100003bfe: 48 8d 3c 03                 	leaq	(%rbx,%rax), %rdi
100003c02: 8b 74 03 20                 	movl	32(%rbx,%rax), %esi
100003c06: 83 ce 05                    	orl	$5, %esi
100003c09: e8 ac 32 00 00              	callq	12972 <dyld_stub_binder+0x100006eba>
100003c0e: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100003c12: e8 8b 32 00 00              	callq	12939 <dyld_stub_binder+0x100006ea2>
100003c17: 48 89 d8                    	movq	%rbx, %rax
100003c1a: 48 83 c4 28                 	addq	$40, %rsp
100003c1e: 5b                          	popq	%rbx
100003c1f: 41 5c                       	popq	%r12
100003c21: 41 5d                       	popq	%r13
100003c23: 41 5e                       	popq	%r14
100003c25: 41 5f                       	popq	%r15
100003c27: 5d                          	popq	%rbp
100003c28: c3                          	retq
100003c29: eb 0e                       	jmp	14 <_main+0x18f9>
100003c2b: 49 89 c6                    	movq	%rax, %r14
100003c2e: 48 8d 7d c8                 	leaq	-56(%rbp), %rdi
100003c32: e8 77 32 00 00              	callq	12919 <dyld_stub_binder+0x100006eae>
100003c37: eb 03                       	jmp	3 <_main+0x18fc>
100003c39: 49 89 c6                    	movq	%rax, %r14
100003c3c: 48 8d 7d b0                 	leaq	-80(%rbp), %rdi
100003c40: e8 5d 32 00 00              	callq	12893 <dyld_stub_binder+0x100006ea2>
100003c45: eb 03                       	jmp	3 <_main+0x190a>
100003c47: 49 89 c6                    	movq	%rax, %r14
100003c4a: 4c 89 f7                    	movq	%r14, %rdi
100003c4d: e8 92 32 00 00              	callq	12946 <dyld_stub_binder+0x100006ee4>
100003c52: 48 8b 03                    	movq	(%rbx), %rax
100003c55: 48 8b 78 e8                 	movq	-24(%rax), %rdi
100003c59: 48 01 df                    	addq	%rbx, %rdi
100003c5c: e8 53 32 00 00              	callq	12883 <dyld_stub_binder+0x100006eb4>
100003c61: e8 84 32 00 00              	callq	12932 <dyld_stub_binder+0x100006eea>
100003c66: eb af                       	jmp	-81 <_main+0x18d7>
100003c68: 48 89 c3                    	movq	%rax, %rbx
100003c6b: e8 7a 32 00 00              	callq	12922 <dyld_stub_binder+0x100006eea>
100003c70: 48 89 df                    	movq	%rbx, %rdi
100003c73: e8 a0 31 00 00              	callq	12704 <dyld_stub_binder+0x100006e18>
100003c78: 0f 0b                       	ud2
100003c7a: 48 89 c7                    	movq	%rax, %rdi
100003c7d: e8 6e fc ff ff              	callq	-914 <_main+0x15b0>
100003c82: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003c8c: 0f 1f 40 00                 	nopl	(%rax)
100003c90: 55                          	pushq	%rbp
100003c91: 48 89 e5                    	movq	%rsp, %rbp
100003c94: 41 57                       	pushq	%r15
100003c96: 41 56                       	pushq	%r14
100003c98: 41 55                       	pushq	%r13
100003c9a: 41 54                       	pushq	%r12
100003c9c: 53                          	pushq	%rbx
100003c9d: 48 83 ec 38                 	subq	$56, %rsp
100003ca1: 48 85 ff                    	testq	%rdi, %rdi
100003ca4: 0f 84 17 01 00 00           	je	279 <_main+0x1a81>
100003caa: 4d 89 c4                    	movq	%r8, %r12
100003cad: 49 89 cf                    	movq	%rcx, %r15
100003cb0: 49 89 fe                    	movq	%rdi, %r14
100003cb3: 44 89 4d bc                 	movl	%r9d, -68(%rbp)
100003cb7: 48 89 c8                    	movq	%rcx, %rax
100003cba: 48 29 f0                    	subq	%rsi, %rax
100003cbd: 49 8b 48 18                 	movq	24(%r8), %rcx
100003cc1: 45 31 ed                    	xorl	%r13d, %r13d
100003cc4: 48 29 c1                    	subq	%rax, %rcx
100003cc7: 4c 0f 4f e9                 	cmovgq	%rcx, %r13
100003ccb: 48 89 55 a8                 	movq	%rdx, -88(%rbp)
100003ccf: 48 89 d3                    	movq	%rdx, %rbx
100003cd2: 48 29 f3                    	subq	%rsi, %rbx
100003cd5: 48 85 db                    	testq	%rbx, %rbx
100003cd8: 7e 15                       	jle	21 <_main+0x19af>
100003cda: 49 8b 06                    	movq	(%r14), %rax
100003cdd: 4c 89 f7                    	movq	%r14, %rdi
100003ce0: 48 89 da                    	movq	%rbx, %rdx
100003ce3: ff 50 60                    	callq	*96(%rax)
100003ce6: 48 39 d8                    	cmpq	%rbx, %rax
100003ce9: 0f 85 d2 00 00 00           	jne	210 <_main+0x1a81>
100003cef: 4d 85 ed                    	testq	%r13, %r13
100003cf2: 0f 8e a1 00 00 00           	jle	161 <_main+0x1a59>
100003cf8: 4c 89 65 b0                 	movq	%r12, -80(%rbp)
100003cfc: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003d00: c5 f8 29 45 c0              	vmovaps	%xmm0, -64(%rbp)
100003d05: 48 c7 45 d0 00 00 00 00     	movq	$0, -48(%rbp)
100003d0d: 49 83 fd 17                 	cmpq	$23, %r13
100003d11: 73 12                       	jae	18 <_main+0x19e5>
100003d13: 43 8d 44 2d 00              	leal	(%r13,%r13), %eax
100003d18: 88 45 c0                    	movb	%al, -64(%rbp)
100003d1b: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
100003d1f: 4c 8d 65 c1                 	leaq	-63(%rbp), %r12
100003d23: eb 27                       	jmp	39 <_main+0x1a0c>
100003d25: 49 8d 5d 10                 	leaq	16(%r13), %rbx
100003d29: 48 83 e3 f0                 	andq	$-16, %rbx
100003d2d: 48 89 df                    	movq	%rbx, %rdi
100003d30: e8 a9 31 00 00              	callq	12713 <dyld_stub_binder+0x100006ede>
100003d35: 49 89 c4                    	movq	%rax, %r12
100003d38: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100003d3c: 48 83 cb 01                 	orq	$1, %rbx
100003d40: 48 89 5d c0                 	movq	%rbx, -64(%rbp)
100003d44: 4c 89 6d c8                 	movq	%r13, -56(%rbp)
100003d48: 48 8d 5d c0                 	leaq	-64(%rbp), %rbx
100003d4c: 0f b6 75 bc                 	movzbl	-68(%rbp), %esi
100003d50: 4c 89 e7                    	movq	%r12, %rdi
100003d53: 4c 89 ea                    	movq	%r13, %rdx
100003d56: e8 9b 31 00 00              	callq	12699 <dyld_stub_binder+0x100006ef6>
100003d5b: 43 c6 04 2c 00              	movb	$0, (%r12,%r13)
100003d60: f6 45 c0 01                 	testb	$1, -64(%rbp)
100003d64: 74 06                       	je	6 <_main+0x1a2c>
100003d66: 48 8b 5d d0                 	movq	-48(%rbp), %rbx
100003d6a: eb 03                       	jmp	3 <_main+0x1a2f>
100003d6c: 48 ff c3                    	incq	%rbx
100003d6f: 4c 8b 65 b0                 	movq	-80(%rbp), %r12
100003d73: 49 8b 06                    	movq	(%r14), %rax
100003d76: 4c 89 f7                    	movq	%r14, %rdi
100003d79: 48 89 de                    	movq	%rbx, %rsi
100003d7c: 4c 89 ea                    	movq	%r13, %rdx
100003d7f: ff 50 60                    	callq	*96(%rax)
100003d82: 48 89 c3                    	movq	%rax, %rbx
100003d85: f6 45 c0 01                 	testb	$1, -64(%rbp)
100003d89: 74 09                       	je	9 <_main+0x1a54>
100003d8b: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
100003d8f: e8 3e 31 00 00              	callq	12606 <dyld_stub_binder+0x100006ed2>
100003d94: 4c 39 eb                    	cmpq	%r13, %rbx
100003d97: 75 28                       	jne	40 <_main+0x1a81>
100003d99: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100003d9d: 49 29 f7                    	subq	%rsi, %r15
100003da0: 4d 85 ff                    	testq	%r15, %r15
100003da3: 7e 11                       	jle	17 <_main+0x1a76>
100003da5: 49 8b 06                    	movq	(%r14), %rax
100003da8: 4c 89 f7                    	movq	%r14, %rdi
100003dab: 4c 89 fa                    	movq	%r15, %rdx
100003dae: ff 50 60                    	callq	*96(%rax)
100003db1: 4c 39 f8                    	cmpq	%r15, %rax
100003db4: 75 0b                       	jne	11 <_main+0x1a81>
100003db6: 49 c7 44 24 18 00 00 00 00  	movq	$0, 24(%r12)
100003dbf: eb 03                       	jmp	3 <_main+0x1a84>
100003dc1: 45 31 f6                    	xorl	%r14d, %r14d
100003dc4: 4c 89 f0                    	movq	%r14, %rax
100003dc7: 48 83 c4 38                 	addq	$56, %rsp
100003dcb: 5b                          	popq	%rbx
100003dcc: 41 5c                       	popq	%r12
100003dce: 41 5d                       	popq	%r13
100003dd0: 41 5e                       	popq	%r14
100003dd2: 41 5f                       	popq	%r15
100003dd4: 5d                          	popq	%rbp
100003dd5: c3                          	retq
100003dd6: 48 89 c3                    	movq	%rax, %rbx
100003dd9: f6 45 c0 01                 	testb	$1, -64(%rbp)
100003ddd: 74 09                       	je	9 <_main+0x1aa8>
100003ddf: 48 8b 7d d0                 	movq	-48(%rbp), %rdi
100003de3: e8 ea 30 00 00              	callq	12522 <dyld_stub_binder+0x100006ed2>
100003de8: 48 89 df                    	movq	%rbx, %rdi
100003deb: e8 28 30 00 00              	callq	12328 <dyld_stub_binder+0x100006e18>
100003df0: 0f 0b                       	ud2
100003df2: 90                          	nop
100003df3: 90                          	nop
100003df4: 90                          	nop
100003df5: 90                          	nop
100003df6: 90                          	nop
100003df7: 90                          	nop
100003df8: 90                          	nop
100003df9: 90                          	nop
100003dfa: 90                          	nop
100003dfb: 90                          	nop
100003dfc: 90                          	nop
100003dfd: 90                          	nop
100003dfe: 90                          	nop
100003dff: 90                          	nop
100003e00: 55                          	pushq	%rbp
100003e01: 48 89 e5                    	movq	%rsp, %rbp
100003e04: 48 8b 05 f5 51 00 00        	movq	20981(%rip), %rax
100003e0b: 80 38 00                    	cmpb	$0, (%rax)
100003e0e: 74 02                       	je	2 <_main+0x1ad2>
100003e10: 5d                          	popq	%rbp
100003e11: c3                          	retq
100003e12: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003e19: 5d                          	popq	%rbp
100003e1a: c3                          	retq
100003e1b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003e20: 55                          	pushq	%rbp
100003e21: 48 89 e5                    	movq	%rsp, %rbp
100003e24: 48 8b 05 f5 51 00 00        	movq	20981(%rip), %rax
100003e2b: 80 38 00                    	cmpb	$0, (%rax)
100003e2e: 74 02                       	je	2 <_main+0x1af2>
100003e30: 5d                          	popq	%rbp
100003e31: c3                          	retq
100003e32: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003e39: 5d                          	popq	%rbp
100003e3a: c3                          	retq
100003e3b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003e40: 55                          	pushq	%rbp
100003e41: 48 89 e5                    	movq	%rsp, %rbp
100003e44: 48 8b 05 ed 51 00 00        	movq	20973(%rip), %rax
100003e4b: 80 38 00                    	cmpb	$0, (%rax)
100003e4e: 74 02                       	je	2 <_main+0x1b12>
100003e50: 5d                          	popq	%rbp
100003e51: c3                          	retq
100003e52: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003e59: 5d                          	popq	%rbp
100003e5a: c3                          	retq
100003e5b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003e60: 55                          	pushq	%rbp
100003e61: 48 89 e5                    	movq	%rsp, %rbp
100003e64: 48 8b 05 c5 51 00 00        	movq	20933(%rip), %rax
100003e6b: 80 38 00                    	cmpb	$0, (%rax)
100003e6e: 74 02                       	je	2 <_main+0x1b32>
100003e70: 5d                          	popq	%rbp
100003e71: c3                          	retq
100003e72: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003e79: 5d                          	popq	%rbp
100003e7a: c3                          	retq
100003e7b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003e80: 55                          	pushq	%rbp
100003e81: 48 89 e5                    	movq	%rsp, %rbp
100003e84: 48 8b 05 9d 51 00 00        	movq	20893(%rip), %rax
100003e8b: 80 38 00                    	cmpb	$0, (%rax)
100003e8e: 74 02                       	je	2 <_main+0x1b52>
100003e90: 5d                          	popq	%rbp
100003e91: c3                          	retq
100003e92: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003e99: 5d                          	popq	%rbp
100003e9a: c3                          	retq
100003e9b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003ea0: 55                          	pushq	%rbp
100003ea1: 48 89 e5                    	movq	%rsp, %rbp
100003ea4: 48 8b 05 5d 51 00 00        	movq	20829(%rip), %rax
100003eab: 80 38 00                    	cmpb	$0, (%rax)
100003eae: 74 02                       	je	2 <_main+0x1b72>
100003eb0: 5d                          	popq	%rbp
100003eb1: c3                          	retq
100003eb2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003eb9: 5d                          	popq	%rbp
100003eba: c3                          	retq
100003ebb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003ec0: 55                          	pushq	%rbp
100003ec1: 48 89 e5                    	movq	%rsp, %rbp
100003ec4: 48 8b 05 45 51 00 00        	movq	20805(%rip), %rax
100003ecb: 80 38 00                    	cmpb	$0, (%rax)
100003ece: 74 02                       	je	2 <_main+0x1b92>
100003ed0: 5d                          	popq	%rbp
100003ed1: c3                          	retq
100003ed2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003ed9: 5d                          	popq	%rbp
100003eda: c3                          	retq
100003edb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003ee0: 55                          	pushq	%rbp
100003ee1: 48 89 e5                    	movq	%rsp, %rbp
100003ee4: 48 8b 05 2d 51 00 00        	movq	20781(%rip), %rax
100003eeb: 80 38 00                    	cmpb	$0, (%rax)
100003eee: 74 02                       	je	2 <_main+0x1bb2>
100003ef0: 5d                          	popq	%rbp
100003ef1: c3                          	retq
100003ef2: 48 c7 00 01 00 00 00        	movq	$1, (%rax)
100003ef9: 5d                          	popq	%rbp
100003efa: c3                          	retq
100003efb: 90                          	nop
100003efc: 90                          	nop
100003efd: 90                          	nop
100003efe: 90                          	nop
100003eff: 90                          	nop

0000000100003f00 __ZN11LineNetworkC2Ev:
100003f00: 55                          	pushq	%rbp
100003f01: 48 89 e5                    	movq	%rsp, %rbp
100003f04: 41 57                       	pushq	%r15
100003f06: 41 56                       	pushq	%r14
100003f08: 41 54                       	pushq	%r12
100003f0a: 53                          	pushq	%rbx
100003f0b: 49 89 fc                    	movq	%rdi, %r12
100003f0e: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
100003f12: c5 f8 11 47 28              	vmovups	%xmm0, 40(%rdi)
100003f17: 48 8d 05 da 51 00 00        	leaq	20954(%rip), %rax
100003f1e: 48 89 07                    	movq	%rax, (%rdi)
100003f21: c6 47 24 00                 	movb	$0, 36(%rdi)
100003f25: bf 00 00 08 00              	movl	$524288, %edi
100003f2a: e8 a9 2f 00 00              	callq	12201 <dyld_stub_binder+0x100006ed8>
100003f2f: 49 89 c7                    	movq	%rax, %r15
100003f32: 49 8d 5c 24 28              	leaq	40(%r12), %rbx
100003f37: 48 89 03                    	movq	%rax, (%rbx)
100003f3a: bf 00 00 08 00              	movl	$524288, %edi
100003f3f: e8 94 2f 00 00              	callq	12180 <dyld_stub_binder+0x100006ed8>
100003f44: 49 89 44 24 30              	movq	%rax, 48(%r12)
100003f49: 66 41 c7 07 00 00           	movw	$0, (%r15)
100003f4f: 31 c0                       	xorl	%eax, %eax
100003f51: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003f5b: 0f 1f 44 00 00              	nopl	(%rax,%rax)
100003f60: 48 8b 0b                    	movq	(%rbx), %rcx
100003f63: c6 44 01 02 00              	movb	$0, 2(%rcx,%rax)
100003f68: 48 8b 0b                    	movq	(%rbx), %rcx
100003f6b: c6 44 01 03 00              	movb	$0, 3(%rcx,%rax)
100003f70: 48 8b 0b                    	movq	(%rbx), %rcx
100003f73: c6 44 01 04 00              	movb	$0, 4(%rcx,%rax)
100003f78: 48 8b 0b                    	movq	(%rbx), %rcx
100003f7b: c6 44 01 05 00              	movb	$0, 5(%rcx,%rax)
100003f80: 48 8b 0b                    	movq	(%rbx), %rcx
100003f83: c6 44 01 06 00              	movb	$0, 6(%rcx,%rax)
100003f88: 48 8b 0b                    	movq	(%rbx), %rcx
100003f8b: c6 44 01 07 00              	movb	$0, 7(%rcx,%rax)
100003f90: 48 8b 0b                    	movq	(%rbx), %rcx
100003f93: c6 44 01 08 00              	movb	$0, 8(%rcx,%rax)
100003f98: 48 8b 0b                    	movq	(%rbx), %rcx
100003f9b: c6 44 01 09 00              	movb	$0, 9(%rcx,%rax)
100003fa0: 48 8b 0b                    	movq	(%rbx), %rcx
100003fa3: c6 44 01 0a 00              	movb	$0, 10(%rcx,%rax)
100003fa8: 48 83 c0 09                 	addq	$9, %rax
100003fac: 48 3d fe ff 07 00           	cmpq	$524286, %rax
100003fb2: 75 ac                       	jne	-84 <__ZN11LineNetworkC2Ev+0x60>
100003fb4: 31 c0                       	xorl	%eax, %eax
100003fb6: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100003fc0: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100003fc5: c6 04 01 00                 	movb	$0, (%rcx,%rax)
100003fc9: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100003fce: c6 44 01 01 00              	movb	$0, 1(%rcx,%rax)
100003fd3: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100003fd8: c6 44 01 02 00              	movb	$0, 2(%rcx,%rax)
100003fdd: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100003fe2: c6 44 01 03 00              	movb	$0, 3(%rcx,%rax)
100003fe7: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100003fec: c6 44 01 04 00              	movb	$0, 4(%rcx,%rax)
100003ff1: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100003ff6: c6 44 01 05 00              	movb	$0, 5(%rcx,%rax)
100003ffb: 49 8b 4c 24 30              	movq	48(%r12), %rcx
100004000: c6 44 01 06 00              	movb	$0, 6(%rcx,%rax)
100004005: 49 8b 4c 24 30              	movq	48(%r12), %rcx
10000400a: c6 44 01 07 00              	movb	$0, 7(%rcx,%rax)
10000400f: 48 83 c0 08                 	addq	$8, %rax
100004013: 48 3d 00 00 08 00           	cmpq	$524288, %rax
100004019: 75 a5                       	jne	-91 <__ZN11LineNetworkC2Ev+0xc0>
10000401b: 41 c7 44 24 20 00 04 a2 01  	movl	$27395072, 32(%r12)
100004024: c5 f8 28 05 34 31 00 00     	vmovaps	12596(%rip), %xmm0
10000402c: c4 c1 78 11 44 24 08        	vmovups	%xmm0, 8(%r12)
100004033: 48 b8 20 00 00 00 20 00 00 00       	movabsq	$137438953504, %rax
10000403d: 49 89 44 24 18              	movq	%rax, 24(%r12)
100004042: 5b                          	popq	%rbx
100004043: 41 5c                       	popq	%r12
100004045: 41 5e                       	popq	%r14
100004047: 41 5f                       	popq	%r15
100004049: 5d                          	popq	%rbp
10000404a: c3                          	retq
10000404b: 49 89 c6                    	movq	%rax, %r14
10000404e: 48 8b 05 fb 4f 00 00        	movq	20475(%rip), %rax
100004055: 48 83 c0 10                 	addq	$16, %rax
100004059: 49 89 04 24                 	movq	%rax, (%r12)
10000405d: 4c 89 ff                    	movq	%r15, %rdi
100004060: e8 6d 2e 00 00              	callq	11885 <dyld_stub_binder+0x100006ed2>
100004065: 49 8b 7c 24 30              	movq	48(%r12), %rdi
10000406a: 48 85 ff                    	testq	%rdi, %rdi
10000406d: 74 21                       	je	33 <__ZN11LineNetworkC2Ev+0x190>
10000406f: e8 5e 2e 00 00              	callq	11870 <dyld_stub_binder+0x100006ed2>
100004074: 4c 89 f7                    	movq	%r14, %rdi
100004077: e8 9c 2d 00 00              	callq	11676 <dyld_stub_binder+0x100006e18>
10000407c: 0f 0b                       	ud2
10000407e: 49 89 c6                    	movq	%rax, %r14
100004081: 48 8b 05 c8 4f 00 00        	movq	20424(%rip), %rax
100004088: 48 83 c0 10                 	addq	$16, %rax
10000408c: 49 89 04 24                 	movq	%rax, (%r12)
100004090: 4c 89 f7                    	movq	%r14, %rdi
100004093: e8 80 2d 00 00              	callq	11648 <dyld_stub_binder+0x100006e18>
100004098: 0f 0b                       	ud2
10000409a: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)

00000001000040a0 __ZN11LineNetworkC1Ev:
1000040a0: 55                          	pushq	%rbp
1000040a1: 48 89 e5                    	movq	%rsp, %rbp
1000040a4: 5d                          	popq	%rbp
1000040a5: e9 56 fe ff ff              	jmp	-426 <__ZN11LineNetworkC2Ev>
1000040aa: 66 0f 1f 44 00 00           	nopw	(%rax,%rax)

00000001000040b0 __ZN11LineNetwork7forwardEv:
1000040b0: 55                          	pushq	%rbp
1000040b1: 48 89 e5                    	movq	%rsp, %rbp
1000040b4: 41 57                       	pushq	%r15
1000040b6: 41 56                       	pushq	%r14
1000040b8: 41 55                       	pushq	%r13
1000040ba: 41 54                       	pushq	%r12
1000040bc: 53                          	pushq	%rbx
1000040bd: 48 83 ec 58                 	subq	$88, %rsp
1000040c1: 48 89 fb                    	movq	%rdi, %rbx
1000040c4: 0f b6 47 24                 	movzbl	36(%rdi), %eax
1000040c8: 31 c9                       	xorl	%ecx, %ecx
1000040ca: 48 85 c0                    	testq	%rax, %rax
1000040cd: 0f 94 c1                    	sete	%cl
1000040d0: 48 8b 7c cf 28              	movq	40(%rdi,%rcx,8), %rdi
1000040d5: 48 8b 74 c3 28              	movq	40(%rbx,%rax,8), %rsi
1000040da: 48 8d 15 bf 32 00 00        	leaq	12991(%rip), %rdx
1000040e1: 48 8d 0d 00 33 00 00        	leaq	13056(%rip), %rcx
1000040e8: 41 b8 37 00 00 00           	movl	$55, %r8d
1000040ee: e8 4d 1b 00 00              	callq	6989 <__ZN11LineNetwork7forwardEv+0x1b90>
1000040f3: 0f b6 4b 24                 	movzbl	36(%rbx), %ecx
1000040f7: 48 83 f1 01                 	xorq	$1, %rcx
1000040fb: 88 4b 24                    	movb	%cl, 36(%rbx)
1000040fe: 31 c0                       	xorl	%eax, %eax
100004100: 84 c9                       	testb	%cl, %cl
100004102: 0f 94 c0                    	sete	%al
100004105: 48 8b 44 c3 28              	movq	40(%rbx,%rax,8), %rax
10000410a: 48 8b 4c cb 28              	movq	40(%rbx,%rcx,8), %rcx
10000410f: 48 8d 91 00 00 08 00        	leaq	524288(%rcx), %rdx
100004116: 48 39 d0                    	cmpq	%rdx, %rax
100004119: 0f 83 b4 00 00 00           	jae	180 <__ZN11LineNetwork7forwardEv+0x123>
10000411f: 48 8d 90 00 00 08 00        	leaq	524288(%rax), %rdx
100004126: 48 39 d1                    	cmpq	%rdx, %rcx
100004129: 0f 83 a4 00 00 00           	jae	164 <__ZN11LineNetwork7forwardEv+0x123>
10000412f: 31 d2                       	xorl	%edx, %edx
100004131: 31 f6                       	xorl	%esi, %esi
100004133: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000413d: 0f 1f 00                    	nopl	(%rax)
100004140: 0f b6 3c 31                 	movzbl	(%rcx,%rsi), %edi
100004144: 40 84 ff                    	testb	%dil, %dil
100004147: 0f 48 fa                    	cmovsl	%edx, %edi
10000414a: 40 88 3c 30                 	movb	%dil, (%rax,%rsi)
10000414e: 0f b6 7c 31 01              	movzbl	1(%rcx,%rsi), %edi
100004153: 40 84 ff                    	testb	%dil, %dil
100004156: 0f 48 fa                    	cmovsl	%edx, %edi
100004159: 40 88 7c 30 01              	movb	%dil, 1(%rax,%rsi)
10000415e: 0f b6 7c 31 02              	movzbl	2(%rcx,%rsi), %edi
100004163: 40 84 ff                    	testb	%dil, %dil
100004166: 0f 48 fa                    	cmovsl	%edx, %edi
100004169: 40 88 7c 30 02              	movb	%dil, 2(%rax,%rsi)
10000416e: 0f b6 7c 31 03              	movzbl	3(%rcx,%rsi), %edi
100004173: 40 84 ff                    	testb	%dil, %dil
100004176: 0f 48 fa                    	cmovsl	%edx, %edi
100004179: 40 88 7c 30 03              	movb	%dil, 3(%rax,%rsi)
10000417e: 0f b6 7c 31 04              	movzbl	4(%rcx,%rsi), %edi
100004183: 40 84 ff                    	testb	%dil, %dil
100004186: 0f 48 fa                    	cmovsl	%edx, %edi
100004189: 40 88 7c 30 04              	movb	%dil, 4(%rax,%rsi)
10000418e: 0f b6 7c 31 05              	movzbl	5(%rcx,%rsi), %edi
100004193: 40 84 ff                    	testb	%dil, %dil
100004196: 0f 48 fa                    	cmovsl	%edx, %edi
100004199: 40 88 7c 30 05              	movb	%dil, 5(%rax,%rsi)
10000419e: 0f b6 7c 31 06              	movzbl	6(%rcx,%rsi), %edi
1000041a3: 40 84 ff                    	testb	%dil, %dil
1000041a6: 0f 48 fa                    	cmovsl	%edx, %edi
1000041a9: 40 88 7c 30 06              	movb	%dil, 6(%rax,%rsi)
1000041ae: 0f b6 7c 31 07              	movzbl	7(%rcx,%rsi), %edi
1000041b3: 40 84 ff                    	testb	%dil, %dil
1000041b6: 0f 48 fa                    	cmovsl	%edx, %edi
1000041b9: 40 88 7c 30 07              	movb	%dil, 7(%rax,%rsi)
1000041be: 48 83 c6 08                 	addq	$8, %rsi
1000041c2: 81 fe 00 00 08 00           	cmpl	$524288, %esi
1000041c8: 0f 85 72 ff ff ff           	jne	-142 <__ZN11LineNetwork7forwardEv+0x90>
1000041ce: e9 c6 04 00 00              	jmp	1222 <__ZN11LineNetwork7forwardEv+0x5e9>
1000041d3: 31 d2                       	xorl	%edx, %edx
1000041d5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000041df: 90                          	nop
1000041e0: c5 7a 6f 34 91              	vmovdqu	(%rcx,%rdx,4), %xmm14
1000041e5: c5 7a 6f 7c 91 10           	vmovdqu	16(%rcx,%rdx,4), %xmm15
1000041eb: c5 fa 6f 54 91 20           	vmovdqu	32(%rcx,%rdx,4), %xmm2
1000041f1: c5 fa 6f 5c 91 30           	vmovdqu	48(%rcx,%rdx,4), %xmm3
1000041f7: c5 79 6f 1d 71 2f 00 00     	vmovdqa	12145(%rip), %xmm11
1000041ff: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100004204: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
100004209: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
10000420d: c5 79 6f 05 6b 2f 00 00     	vmovdqa	12139(%rip), %xmm8
100004215: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
10000421a: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
10000421f: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004223: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
100004229: c5 fa 6f 64 91 70           	vmovdqu	112(%rcx,%rdx,4), %xmm4
10000422f: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100004234: c4 e3 fd 00 6c 91 60 4e     	vpermq	$78, 96(%rcx,%rdx,4), %ymm5
10000423c: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004242: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004247: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000424b: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100004251: c5 fa 6f 74 91 50           	vmovdqu	80(%rcx,%rdx,4), %xmm6
100004257: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
10000425c: c4 63 fd 00 4c 91 40 4e     	vpermq	$78, 64(%rcx,%rdx,4), %ymm9
100004264: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
10000426a: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
10000426f: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100004274: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000427a: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100004280: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004286: c5 79 6f 05 02 2f 00 00     	vmovdqa	12034(%rip), %xmm8
10000428e: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100004293: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100004298: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
10000429c: c5 79 6f 1d fc 2e 00 00     	vmovdqa	12028(%rip), %xmm11
1000042a4: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
1000042a9: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
1000042ae: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
1000042b2: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
1000042b8: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
1000042bd: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
1000042c2: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000042c6: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000042cc: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
1000042d1: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
1000042d6: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
1000042da: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000042e0: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000042e6: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
1000042ec: c5 79 6f 1d bc 2e 00 00     	vmovdqa	11964(%rip), %xmm11
1000042f4: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
1000042f9: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
1000042fe: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
100004302: c5 f9 6f 0d b6 2e 00 00     	vmovdqa	11958(%rip), %xmm1
10000430a: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
10000430e: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100004313: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100004318: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
10000431c: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100004322: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100004327: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
10000432c: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004330: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004336: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
10000433b: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100004340: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100004344: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000434a: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004350: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100004356: c5 f9 6f 0d 72 2e 00 00     	vmovdqa	11890(%rip), %xmm1
10000435e: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100004363: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100004368: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000436c: c5 f9 6f 15 6c 2e 00 00     	vmovdqa	11884(%rip), %xmm2
100004374: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100004378: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
10000437d: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100004382: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004386: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
10000438c: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100004391: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100004396: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000439a: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000043a0: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
1000043a5: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
1000043aa: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000043ae: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
1000043b4: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
1000043ba: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
1000043c0: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
1000043c5: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
1000043ca: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
1000043cf: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
1000043d4: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000043d9: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
1000043dd: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000043e1: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
1000043e5: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
1000043e9: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000043ed: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000043f1: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000043f5: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000043f9: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
1000043ff: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
100004405: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
10000440b: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100004411: c5 fe 7f 4c 90 40           	vmovdqu	%ymm1, 64(%rax,%rdx,4)
100004417: c5 fe 7f 44 90 60           	vmovdqu	%ymm0, 96(%rax,%rdx,4)
10000441d: c5 fe 7f 6c 90 20           	vmovdqu	%ymm5, 32(%rax,%rdx,4)
100004423: c5 fe 7f 14 90              	vmovdqu	%ymm2, (%rax,%rdx,4)
100004428: c5 7a 6f a4 91 80 00 00 00  	vmovdqu	128(%rcx,%rdx,4), %xmm12
100004431: c5 7a 6f ac 91 90 00 00 00  	vmovdqu	144(%rcx,%rdx,4), %xmm13
10000443a: c5 7a 6f b4 91 a0 00 00 00  	vmovdqu	160(%rcx,%rdx,4), %xmm14
100004443: c5 fa 6f 9c 91 b0 00 00 00  	vmovdqu	176(%rcx,%rdx,4), %xmm3
10000444c: c5 f9 6f 05 1c 2d 00 00     	vmovdqa	11548(%rip), %xmm0
100004454: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
100004459: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
10000445e: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100004462: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100004466: c5 f9 6f 05 12 2d 00 00     	vmovdqa	11538(%rip), %xmm0
10000446e: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100004473: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
100004478: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
10000447c: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004480: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
100004486: c5 fa 6f a4 91 f0 00 00 00  	vmovdqu	240(%rcx,%rdx,4), %xmm4
10000448f: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100004494: c4 e3 fd 00 ac 91 e0 00 00 00 4e    	vpermq	$78, 224(%rcx,%rdx,4), %ymm5
10000449f: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
1000044a5: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
1000044aa: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
1000044ae: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
1000044b4: c5 fa 6f b4 91 d0 00 00 00  	vmovdqu	208(%rcx,%rdx,4), %xmm6
1000044bd: c4 e3 fd 00 bc 91 c0 00 00 00 4e    	vpermq	$78, 192(%rcx,%rdx,4), %ymm7
1000044c8: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
1000044cd: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000044d3: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
1000044d8: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
1000044dc: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000044e2: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
1000044e8: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
1000044ee: c5 79 6f 3d 9a 2c 00 00     	vmovdqa	11418(%rip), %xmm15
1000044f6: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
1000044fb: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
100004500: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
100004504: c5 f9 6f 05 94 2c 00 00     	vmovdqa	11412(%rip), %xmm0
10000450c: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100004511: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100004516: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000451a: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
100004520: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
100004525: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
10000452a: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000452e: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004534: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100004539: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
10000453e: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100004542: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004548: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
10000454e: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004554: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100004559: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000455e: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100004562: c5 f9 6f 05 56 2c 00 00     	vmovdqa	11350(%rip), %xmm0
10000456a: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000456f: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100004574: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004578: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
10000457e: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100004583: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100004588: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000458c: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100004591: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100004596: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
10000459a: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000045a0: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000045a6: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000045ac: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
1000045b2: c5 79 6f 3d 16 2c 00 00     	vmovdqa	11286(%rip), %xmm15
1000045ba: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
1000045bf: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
1000045c4: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000045c8: c5 f9 6f 05 10 2c 00 00     	vmovdqa	11280(%rip), %xmm0
1000045d0: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
1000045d5: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
1000045da: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000045de: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
1000045e4: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
1000045e9: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
1000045ee: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000045f2: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
1000045f7: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
1000045fc: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100004600: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100004606: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
10000460c: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100004612: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
100004618: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
10000461d: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
100004622: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
100004627: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
10000462c: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100004630: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100004634: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100004638: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
10000463c: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100004640: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100004644: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100004648: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
10000464c: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100004652: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100004658: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
10000465e: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100004664: c5 fe 7f 8c 90 c0 00 00 00  	vmovdqu	%ymm1, 192(%rax,%rdx,4)
10000466d: c5 fe 7f 84 90 e0 00 00 00  	vmovdqu	%ymm0, 224(%rax,%rdx,4)
100004676: c5 fe 7f 9c 90 a0 00 00 00  	vmovdqu	%ymm3, 160(%rax,%rdx,4)
10000467f: c5 fe 7f 94 90 80 00 00 00  	vmovdqu	%ymm2, 128(%rax,%rdx,4)
100004688: 48 83 c2 40                 	addq	$64, %rdx
10000468c: 48 81 fa 00 00 02 00        	cmpq	$131072, %rdx
100004693: 0f 85 47 fb ff ff           	jne	-1209 <__ZN11LineNetwork7forwardEv+0x130>
100004699: 0f b6 43 24                 	movzbl	36(%rbx), %eax
10000469d: 48 83 f0 01                 	xorq	$1, %rax
1000046a1: 88 43 24                    	movb	%al, 36(%rbx)
1000046a4: 31 c9                       	xorl	%ecx, %ecx
1000046a6: 84 c0                       	testb	%al, %al
1000046a8: 0f 94 c1                    	sete	%cl
1000046ab: 4c 8b 64 cb 28              	movq	40(%rbx,%rcx,8), %r12
1000046b0: 48 89 5d 90                 	movq	%rbx, -112(%rbp)
1000046b4: 48 8b 44 c3 28              	movq	40(%rbx,%rax,8), %rax
1000046b9: 48 89 45 c8                 	movq	%rax, -56(%rbp)
1000046bd: 31 c0                       	xorl	%eax, %eax
1000046bf: eb 27                       	jmp	39 <__ZN11LineNetwork7forwardEv+0x638>
1000046c1: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000046cb: 0f 1f 44 00 00              	nopl	(%rax,%rax)
1000046d0: 48 8b 45 d0                 	movq	-48(%rbp), %rax
1000046d4: 48 ff c0                    	incq	%rax
1000046d7: 4c 8b 65 c0                 	movq	-64(%rbp), %r12
1000046db: 49 ff c4                    	incq	%r12
1000046de: 48 83 f8 08                 	cmpq	$8, %rax
1000046e2: 0f 84 04 01 00 00           	je	260 <__ZN11LineNetwork7forwardEv+0x73c>
1000046e8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
1000046ec: 48 8d 04 c0                 	leaq	(%rax,%rax,8), %rax
1000046f0: 48 8d 0d f9 2c 00 00        	leaq	11513(%rip), %rcx
1000046f7: 48 8d 14 c1                 	leaq	(%rcx,%rax,8), %rdx
1000046fb: 48 89 55 98                 	movq	%rdx, -104(%rbp)
1000046ff: 48 8d 54 c1 18              	leaq	24(%rcx,%rax,8), %rdx
100004704: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100004708: 48 8d 44 c1 30              	leaq	48(%rcx,%rax,8), %rax
10000470d: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100004711: 4c 89 65 c0                 	movq	%r12, -64(%rbp)
100004715: 48 8b 5d c8                 	movq	-56(%rbp), %rbx
100004719: 31 c0                       	xorl	%eax, %eax
10000471b: eb 21                       	jmp	33 <__ZN11LineNetwork7forwardEv+0x68e>
10000471d: 0f 1f 00                    	nopl	(%rax)
100004720: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100004724: 48 81 c3 00 10 00 00        	addq	$4096, %rbx
10000472b: 49 81 c4 00 04 00 00        	addq	$1024, %r12
100004732: 48 8b 45 b8                 	movq	-72(%rbp), %rax
100004736: 48 3d fd 00 00 00           	cmpq	$253, %rax
10000473c: 73 92                       	jae	-110 <__ZN11LineNetwork7forwardEv+0x620>
10000473e: 48 83 c0 02                 	addq	$2, %rax
100004742: 48 89 45 b8                 	movq	%rax, -72(%rbp)
100004746: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
10000474a: 45 31 ed                    	xorl	%r13d, %r13d
10000474d: eb 16                       	jmp	22 <__ZN11LineNetwork7forwardEv+0x6b5>
10000474f: 90                          	nop
100004750: 43 88 04 ac                 	movb	%al, (%r12,%r13,4)
100004754: 49 83 c5 02                 	addq	$2, %r13
100004758: 48 83 c3 10                 	addq	$16, %rbx
10000475c: 49 81 fd fd 00 00 00        	cmpq	$253, %r13
100004763: 73 bb                       	jae	-69 <__ZN11LineNetwork7forwardEv+0x670>
100004765: 48 89 df                    	movq	%rbx, %rdi
100004768: 48 8b 75 98                 	movq	-104(%rbp), %rsi
10000476c: c5 f8 77                    	vzeroupper
10000476f: e8 ec 22 00 00              	callq	8940 <__ZN11LineNetwork7forwardEv+0x29b0>
100004774: 41 89 c6                    	movl	%eax, %r14d
100004777: 48 8d bb 00 08 00 00        	leaq	2048(%rbx), %rdi
10000477e: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100004782: e8 d9 22 00 00              	callq	8921 <__ZN11LineNetwork7forwardEv+0x29b0>
100004787: 41 89 c7                    	movl	%eax, %r15d
10000478a: 45 01 f7                    	addl	%r14d, %r15d
10000478d: 48 8d bb 00 10 00 00        	leaq	4096(%rbx), %rdi
100004794: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100004798: e8 c3 22 00 00              	callq	8899 <__ZN11LineNetwork7forwardEv+0x29b0>
10000479d: 44 01 f8                    	addl	%r15d, %eax
1000047a0: 48 8b 4d d0                 	movq	-48(%rbp), %rcx
1000047a4: 48 8d 15 85 2e 00 00        	leaq	11909(%rip), %rdx
1000047ab: 0f be 0c 11                 	movsbl	(%rcx,%rdx), %ecx
1000047af: 01 c1                       	addl	%eax, %ecx
1000047b1: 6b c1 37                    	imull	$55, %ecx, %eax
1000047b4: 48 98                       	cltq
1000047b6: 48 69 c8 09 04 02 81        	imulq	$-2130574327, %rax, %rcx
1000047bd: 48 c1 e9 20                 	shrq	$32, %rcx
1000047c1: 01 c8                       	addl	%ecx, %eax
1000047c3: 89 c1                       	movl	%eax, %ecx
1000047c5: c1 e9 1f                    	shrl	$31, %ecx
1000047c8: c1 f8 0d                    	sarl	$13, %eax
1000047cb: 01 c8                       	addl	%ecx, %eax
1000047cd: 3d 80 00 00 00              	cmpl	$128, %eax
1000047d2: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x729>
1000047d4: b8 7f 00 00 00              	movl	$127, %eax
1000047d9: 83 f8 81                    	cmpl	$-127, %eax
1000047dc: 0f 8f 6e ff ff ff           	jg	-146 <__ZN11LineNetwork7forwardEv+0x6a0>
1000047e2: b8 81 00 00 00              	movl	$129, %eax
1000047e7: e9 64 ff ff ff              	jmp	-156 <__ZN11LineNetwork7forwardEv+0x6a0>
1000047ec: 48 8b 5d 90                 	movq	-112(%rbp), %rbx
1000047f0: 0f b6 4b 24                 	movzbl	36(%rbx), %ecx
1000047f4: 48 83 f1 01                 	xorq	$1, %rcx
1000047f8: 88 4b 24                    	movb	%cl, 36(%rbx)
1000047fb: 31 c0                       	xorl	%eax, %eax
1000047fd: 84 c9                       	testb	%cl, %cl
1000047ff: 0f 94 c0                    	sete	%al
100004802: 48 8b 44 c3 28              	movq	40(%rbx,%rax,8), %rax
100004807: 48 8b 4c cb 28              	movq	40(%rbx,%rcx,8), %rcx
10000480c: 48 8d 91 00 00 02 00        	leaq	131072(%rcx), %rdx
100004813: 48 39 d0                    	cmpq	%rdx, %rax
100004816: 0f 83 a7 00 00 00           	jae	167 <__ZN11LineNetwork7forwardEv+0x813>
10000481c: 48 8d 90 00 00 02 00        	leaq	131072(%rax), %rdx
100004823: 48 39 d1                    	cmpq	%rdx, %rcx
100004826: 0f 83 97 00 00 00           	jae	151 <__ZN11LineNetwork7forwardEv+0x813>
10000482c: 31 d2                       	xorl	%edx, %edx
10000482e: 31 f6                       	xorl	%esi, %esi
100004830: 0f b6 3c 31                 	movzbl	(%rcx,%rsi), %edi
100004834: 40 84 ff                    	testb	%dil, %dil
100004837: 0f 48 fa                    	cmovsl	%edx, %edi
10000483a: 40 88 3c 30                 	movb	%dil, (%rax,%rsi)
10000483e: 0f b6 7c 31 01              	movzbl	1(%rcx,%rsi), %edi
100004843: 40 84 ff                    	testb	%dil, %dil
100004846: 0f 48 fa                    	cmovsl	%edx, %edi
100004849: 40 88 7c 30 01              	movb	%dil, 1(%rax,%rsi)
10000484e: 0f b6 7c 31 02              	movzbl	2(%rcx,%rsi), %edi
100004853: 40 84 ff                    	testb	%dil, %dil
100004856: 0f 48 fa                    	cmovsl	%edx, %edi
100004859: 40 88 7c 30 02              	movb	%dil, 2(%rax,%rsi)
10000485e: 0f b6 7c 31 03              	movzbl	3(%rcx,%rsi), %edi
100004863: 40 84 ff                    	testb	%dil, %dil
100004866: 0f 48 fa                    	cmovsl	%edx, %edi
100004869: 40 88 7c 30 03              	movb	%dil, 3(%rax,%rsi)
10000486e: 0f b6 7c 31 04              	movzbl	4(%rcx,%rsi), %edi
100004873: 40 84 ff                    	testb	%dil, %dil
100004876: 0f 48 fa                    	cmovsl	%edx, %edi
100004879: 40 88 7c 30 04              	movb	%dil, 4(%rax,%rsi)
10000487e: 0f b6 7c 31 05              	movzbl	5(%rcx,%rsi), %edi
100004883: 40 84 ff                    	testb	%dil, %dil
100004886: 0f 48 fa                    	cmovsl	%edx, %edi
100004889: 40 88 7c 30 05              	movb	%dil, 5(%rax,%rsi)
10000488e: 0f b6 7c 31 06              	movzbl	6(%rcx,%rsi), %edi
100004893: 40 84 ff                    	testb	%dil, %dil
100004896: 0f 48 fa                    	cmovsl	%edx, %edi
100004899: 40 88 7c 30 06              	movb	%dil, 6(%rax,%rsi)
10000489e: 0f b6 7c 31 07              	movzbl	7(%rcx,%rsi), %edi
1000048a3: 40 84 ff                    	testb	%dil, %dil
1000048a6: 0f 48 fa                    	cmovsl	%edx, %edi
1000048a9: 40 88 7c 30 07              	movb	%dil, 7(%rax,%rsi)
1000048ae: 48 83 c6 08                 	addq	$8, %rsi
1000048b2: 81 fe 00 00 02 00           	cmpl	$131072, %esi
1000048b8: 0f 85 72 ff ff ff           	jne	-142 <__ZN11LineNetwork7forwardEv+0x780>
1000048be: e9 c6 04 00 00              	jmp	1222 <__ZN11LineNetwork7forwardEv+0xcd9>
1000048c3: 31 d2                       	xorl	%edx, %edx
1000048c5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000048cf: 90                          	nop
1000048d0: c5 7a 6f 34 91              	vmovdqu	(%rcx,%rdx,4), %xmm14
1000048d5: c5 7a 6f 7c 91 10           	vmovdqu	16(%rcx,%rdx,4), %xmm15
1000048db: c5 fa 6f 54 91 20           	vmovdqu	32(%rcx,%rdx,4), %xmm2
1000048e1: c5 fa 6f 5c 91 30           	vmovdqu	48(%rcx,%rdx,4), %xmm3
1000048e7: c5 79 6f 1d 81 28 00 00     	vmovdqa	10369(%rip), %xmm11
1000048ef: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
1000048f4: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
1000048f9: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
1000048fd: c5 79 6f 05 7b 28 00 00     	vmovdqa	10363(%rip), %xmm8
100004905: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
10000490a: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
10000490f: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004913: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
100004919: c5 fa 6f 64 91 70           	vmovdqu	112(%rcx,%rdx,4), %xmm4
10000491f: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100004924: c4 e3 fd 00 6c 91 60 4e     	vpermq	$78, 96(%rcx,%rdx,4), %ymm5
10000492c: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004932: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004937: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000493b: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100004941: c5 fa 6f 74 91 50           	vmovdqu	80(%rcx,%rdx,4), %xmm6
100004947: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
10000494c: c4 63 fd 00 4c 91 40 4e     	vpermq	$78, 64(%rcx,%rdx,4), %ymm9
100004954: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
10000495a: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
10000495f: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100004964: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000496a: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100004970: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004976: c5 79 6f 05 12 28 00 00     	vmovdqa	10258(%rip), %xmm8
10000497e: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100004983: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100004988: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
10000498c: c5 79 6f 1d 0c 28 00 00     	vmovdqa	10252(%rip), %xmm11
100004994: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
100004999: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000499e: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
1000049a2: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
1000049a8: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
1000049ad: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
1000049b2: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000049b6: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000049bc: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
1000049c1: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
1000049c6: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
1000049ca: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000049d0: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000049d6: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
1000049dc: c5 79 6f 1d cc 27 00 00     	vmovdqa	10188(%rip), %xmm11
1000049e4: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
1000049e9: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
1000049ee: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
1000049f2: c5 f9 6f 0d c6 27 00 00     	vmovdqa	10182(%rip), %xmm1
1000049fa: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
1000049fe: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
100004a03: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
100004a08: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004a0c: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100004a12: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100004a17: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100004a1c: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100004a20: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004a26: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
100004a2b: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100004a30: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100004a34: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004a3a: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004a40: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100004a46: c5 f9 6f 0d 82 27 00 00     	vmovdqa	10114(%rip), %xmm1
100004a4e: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100004a53: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100004a58: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
100004a5c: c5 f9 6f 15 7c 27 00 00     	vmovdqa	10108(%rip), %xmm2
100004a64: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100004a68: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
100004a6d: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100004a72: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004a76: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
100004a7c: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100004a81: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100004a86: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004a8a: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100004a90: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
100004a95: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
100004a9a: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100004a9e: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100004aa4: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100004aaa: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
100004ab0: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
100004ab5: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
100004aba: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
100004abf: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
100004ac4: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100004ac9: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100004acd: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100004ad1: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100004ad5: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100004ad9: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100004add: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100004ae1: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100004ae5: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100004ae9: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100004aef: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
100004af5: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100004afb: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100004b01: c5 fe 7f 4c 90 40           	vmovdqu	%ymm1, 64(%rax,%rdx,4)
100004b07: c5 fe 7f 44 90 60           	vmovdqu	%ymm0, 96(%rax,%rdx,4)
100004b0d: c5 fe 7f 6c 90 20           	vmovdqu	%ymm5, 32(%rax,%rdx,4)
100004b13: c5 fe 7f 14 90              	vmovdqu	%ymm2, (%rax,%rdx,4)
100004b18: c5 7a 6f a4 91 80 00 00 00  	vmovdqu	128(%rcx,%rdx,4), %xmm12
100004b21: c5 7a 6f ac 91 90 00 00 00  	vmovdqu	144(%rcx,%rdx,4), %xmm13
100004b2a: c5 7a 6f b4 91 a0 00 00 00  	vmovdqu	160(%rcx,%rdx,4), %xmm14
100004b33: c5 fa 6f 9c 91 b0 00 00 00  	vmovdqu	176(%rcx,%rdx,4), %xmm3
100004b3c: c5 f9 6f 05 2c 26 00 00     	vmovdqa	9772(%rip), %xmm0
100004b44: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
100004b49: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
100004b4e: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100004b52: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100004b56: c5 f9 6f 05 22 26 00 00     	vmovdqa	9762(%rip), %xmm0
100004b5e: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100004b63: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
100004b68: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
100004b6c: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004b70: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
100004b76: c5 fa 6f a4 91 f0 00 00 00  	vmovdqu	240(%rcx,%rdx,4), %xmm4
100004b7f: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100004b84: c4 e3 fd 00 ac 91 e0 00 00 00 4e    	vpermq	$78, 224(%rcx,%rdx,4), %ymm5
100004b8f: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100004b95: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
100004b9a: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
100004b9e: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
100004ba4: c5 fa 6f b4 91 d0 00 00 00  	vmovdqu	208(%rcx,%rdx,4), %xmm6
100004bad: c4 e3 fd 00 bc 91 c0 00 00 00 4e    	vpermq	$78, 192(%rcx,%rdx,4), %ymm7
100004bb8: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
100004bbd: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
100004bc3: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
100004bc8: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
100004bcc: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004bd2: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
100004bd8: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
100004bde: c5 79 6f 3d aa 25 00 00     	vmovdqa	9642(%rip), %xmm15
100004be6: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
100004beb: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
100004bf0: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
100004bf4: c5 f9 6f 05 a4 25 00 00     	vmovdqa	9636(%rip), %xmm0
100004bfc: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100004c01: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100004c06: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004c0a: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
100004c10: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
100004c15: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
100004c1a: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004c1e: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004c24: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100004c29: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100004c2e: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100004c32: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004c38: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004c3e: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100004c44: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100004c49: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100004c4e: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100004c52: c5 f9 6f 05 66 25 00 00     	vmovdqa	9574(%rip), %xmm0
100004c5a: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100004c5f: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100004c64: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004c68: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
100004c6e: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100004c73: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100004c78: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004c7c: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100004c81: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100004c86: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100004c8a: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100004c90: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100004c96: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100004c9c: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
100004ca2: c5 79 6f 3d 26 25 00 00     	vmovdqa	9510(%rip), %xmm15
100004caa: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
100004caf: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
100004cb4: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100004cb8: c5 f9 6f 05 20 25 00 00     	vmovdqa	9504(%rip), %xmm0
100004cc0: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
100004cc5: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
100004cca: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004cce: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
100004cd4: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
100004cd9: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
100004cde: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100004ce2: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
100004ce7: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
100004cec: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100004cf0: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100004cf6: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100004cfc: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100004d02: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
100004d08: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
100004d0d: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
100004d12: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
100004d17: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100004d1c: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100004d20: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100004d24: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100004d28: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100004d2c: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100004d30: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100004d34: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100004d38: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100004d3c: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100004d42: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100004d48: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
100004d4e: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100004d54: c5 fe 7f 8c 90 c0 00 00 00  	vmovdqu	%ymm1, 192(%rax,%rdx,4)
100004d5d: c5 fe 7f 84 90 e0 00 00 00  	vmovdqu	%ymm0, 224(%rax,%rdx,4)
100004d66: c5 fe 7f 9c 90 a0 00 00 00  	vmovdqu	%ymm3, 160(%rax,%rdx,4)
100004d6f: c5 fe 7f 94 90 80 00 00 00  	vmovdqu	%ymm2, 128(%rax,%rdx,4)
100004d78: 48 83 c2 40                 	addq	$64, %rdx
100004d7c: 48 81 fa 00 80 00 00        	cmpq	$32768, %rdx
100004d83: 0f 85 47 fb ff ff           	jne	-1209 <__ZN11LineNetwork7forwardEv+0x820>
100004d89: 0f b6 43 24                 	movzbl	36(%rbx), %eax
100004d8d: 48 83 f0 01                 	xorq	$1, %rax
100004d91: 88 43 24                    	movb	%al, 36(%rbx)
100004d94: 31 c9                       	xorl	%ecx, %ecx
100004d96: 84 c0                       	testb	%al, %al
100004d98: 0f 94 c1                    	sete	%cl
100004d9b: 4c 8b 64 cb 28              	movq	40(%rbx,%rcx,8), %r12
100004da0: 48 8b 44 c3 28              	movq	40(%rbx,%rax,8), %rax
100004da5: 48 89 45 c8                 	movq	%rax, -56(%rbp)
100004da9: 31 c0                       	xorl	%eax, %eax
100004dab: eb 1b                       	jmp	27 <__ZN11LineNetwork7forwardEv+0xd18>
100004dad: 0f 1f 00                    	nopl	(%rax)
100004db0: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100004db4: 48 ff c0                    	incq	%rax
100004db7: 4c 8b 65 c0                 	movq	-64(%rbp), %r12
100004dbb: 49 ff c4                    	incq	%r12
100004dbe: 48 83 f8 10                 	cmpq	$16, %rax
100004dc2: 0f 84 01 01 00 00           	je	257 <__ZN11LineNetwork7forwardEv+0xe19>
100004dc8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100004dcc: 48 8d 04 c0                 	leaq	(%rax,%rax,8), %rax
100004dd0: 48 8d 0d 69 28 00 00        	leaq	10345(%rip), %rcx
100004dd7: 48 8d 14 c1                 	leaq	(%rcx,%rax,8), %rdx
100004ddb: 48 89 55 98                 	movq	%rdx, -104(%rbp)
100004ddf: 48 8d 54 c1 18              	leaq	24(%rcx,%rax,8), %rdx
100004de4: 48 89 55 a0                 	movq	%rdx, -96(%rbp)
100004de8: 48 8d 44 c1 30              	leaq	48(%rcx,%rax,8), %rax
100004ded: 48 89 45 a8                 	movq	%rax, -88(%rbp)
100004df1: 4c 89 65 c0                 	movq	%r12, -64(%rbp)
100004df5: 48 8b 5d c8                 	movq	-56(%rbp), %rbx
100004df9: 31 c0                       	xorl	%eax, %eax
100004dfb: eb 1f                       	jmp	31 <__ZN11LineNetwork7forwardEv+0xd6c>
100004dfd: 0f 1f 00                    	nopl	(%rax)
100004e00: 48 8b 5d b0                 	movq	-80(%rbp), %rbx
100004e04: 48 81 c3 00 08 00 00        	addq	$2048, %rbx
100004e0b: 49 81 c4 00 04 00 00        	addq	$1024, %r12
100004e12: 48 8b 45 b8                 	movq	-72(%rbp), %rax
100004e16: 48 83 f8 7d                 	cmpq	$125, %rax
100004e1a: 73 94                       	jae	-108 <__ZN11LineNetwork7forwardEv+0xd00>
100004e1c: 48 83 c0 02                 	addq	$2, %rax
100004e20: 48 89 45 b8                 	movq	%rax, -72(%rbp)
100004e24: 48 89 5d b0                 	movq	%rbx, -80(%rbp)
100004e28: 45 31 ed                    	xorl	%r13d, %r13d
100004e2b: eb 15                       	jmp	21 <__ZN11LineNetwork7forwardEv+0xd92>
100004e2d: 0f 1f 00                    	nopl	(%rax)
100004e30: 43 88 04 ec                 	movb	%al, (%r12,%r13,8)
100004e34: 49 83 c5 02                 	addq	$2, %r13
100004e38: 48 83 c3 10                 	addq	$16, %rbx
100004e3c: 49 83 fd 7d                 	cmpq	$125, %r13
100004e40: 73 be                       	jae	-66 <__ZN11LineNetwork7forwardEv+0xd50>
100004e42: 48 89 df                    	movq	%rbx, %rdi
100004e45: 48 8b 75 98                 	movq	-104(%rbp), %rsi
100004e49: c5 f8 77                    	vzeroupper
100004e4c: e8 0f 1c 00 00              	callq	7183 <__ZN11LineNetwork7forwardEv+0x29b0>
100004e51: 41 89 c6                    	movl	%eax, %r14d
100004e54: 48 8d bb 00 04 00 00        	leaq	1024(%rbx), %rdi
100004e5b: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100004e5f: e8 fc 1b 00 00              	callq	7164 <__ZN11LineNetwork7forwardEv+0x29b0>
100004e64: 41 89 c7                    	movl	%eax, %r15d
100004e67: 45 01 f7                    	addl	%r14d, %r15d
100004e6a: 48 8d bb 00 08 00 00        	leaq	2048(%rbx), %rdi
100004e71: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100004e75: e8 e6 1b 00 00              	callq	7142 <__ZN11LineNetwork7forwardEv+0x29b0>
100004e7a: 44 01 f8                    	addl	%r15d, %eax
100004e7d: 48 8b 4d d0                 	movq	-48(%rbp), %rcx
100004e81: 48 8d 15 38 2c 00 00        	leaq	11320(%rip), %rdx
100004e88: 0f be 0c 11                 	movsbl	(%rcx,%rdx), %ecx
100004e8c: 01 c1                       	addl	%eax, %ecx
100004e8e: 6b c1 39                    	imull	$57, %ecx, %eax
100004e91: 48 98                       	cltq
100004e93: 48 69 c8 09 04 02 81        	imulq	$-2130574327, %rax, %rcx
100004e9a: 48 c1 e9 20                 	shrq	$32, %rcx
100004e9e: 01 c8                       	addl	%ecx, %eax
100004ea0: 89 c1                       	movl	%eax, %ecx
100004ea2: c1 e9 1f                    	shrl	$31, %ecx
100004ea5: c1 f8 0d                    	sarl	$13, %eax
100004ea8: 01 c8                       	addl	%ecx, %eax
100004eaa: 3d 80 00 00 00              	cmpl	$128, %eax
100004eaf: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0xe06>
100004eb1: b8 7f 00 00 00              	movl	$127, %eax
100004eb6: 83 f8 81                    	cmpl	$-127, %eax
100004eb9: 0f 8f 71 ff ff ff           	jg	-143 <__ZN11LineNetwork7forwardEv+0xd80>
100004ebf: b8 81 00 00 00              	movl	$129, %eax
100004ec4: e9 67 ff ff ff              	jmp	-153 <__ZN11LineNetwork7forwardEv+0xd80>
100004ec9: 48 8b 5d 90                 	movq	-112(%rbp), %rbx
100004ecd: 0f b6 4b 24                 	movzbl	36(%rbx), %ecx
100004ed1: 48 83 f1 01                 	xorq	$1, %rcx
100004ed5: 88 4b 24                    	movb	%cl, 36(%rbx)
100004ed8: 31 c0                       	xorl	%eax, %eax
100004eda: 84 c9                       	testb	%cl, %cl
100004edc: 0f 94 c0                    	sete	%al
100004edf: 48 8b 44 c3 28              	movq	40(%rbx,%rax,8), %rax
100004ee4: 48 8b 4c cb 28              	movq	40(%rbx,%rcx,8), %rcx
100004ee9: 48 8d 91 00 00 01 00        	leaq	65536(%rcx), %rdx
100004ef0: 48 39 d0                    	cmpq	%rdx, %rax
100004ef3: 0f 83 aa 00 00 00           	jae	170 <__ZN11LineNetwork7forwardEv+0xef3>
100004ef9: 48 8d 90 00 00 01 00        	leaq	65536(%rax), %rdx
100004f00: 48 39 d1                    	cmpq	%rdx, %rcx
100004f03: 0f 83 9a 00 00 00           	jae	154 <__ZN11LineNetwork7forwardEv+0xef3>
100004f09: 31 d2                       	xorl	%edx, %edx
100004f0b: 31 f6                       	xorl	%esi, %esi
100004f0d: 0f 1f 00                    	nopl	(%rax)
100004f10: 0f b6 3c 31                 	movzbl	(%rcx,%rsi), %edi
100004f14: 40 84 ff                    	testb	%dil, %dil
100004f17: 0f 48 fa                    	cmovsl	%edx, %edi
100004f1a: 40 88 3c 30                 	movb	%dil, (%rax,%rsi)
100004f1e: 0f b6 7c 31 01              	movzbl	1(%rcx,%rsi), %edi
100004f23: 40 84 ff                    	testb	%dil, %dil
100004f26: 0f 48 fa                    	cmovsl	%edx, %edi
100004f29: 40 88 7c 30 01              	movb	%dil, 1(%rax,%rsi)
100004f2e: 0f b6 7c 31 02              	movzbl	2(%rcx,%rsi), %edi
100004f33: 40 84 ff                    	testb	%dil, %dil
100004f36: 0f 48 fa                    	cmovsl	%edx, %edi
100004f39: 40 88 7c 30 02              	movb	%dil, 2(%rax,%rsi)
100004f3e: 0f b6 7c 31 03              	movzbl	3(%rcx,%rsi), %edi
100004f43: 40 84 ff                    	testb	%dil, %dil
100004f46: 0f 48 fa                    	cmovsl	%edx, %edi
100004f49: 40 88 7c 30 03              	movb	%dil, 3(%rax,%rsi)
100004f4e: 0f b6 7c 31 04              	movzbl	4(%rcx,%rsi), %edi
100004f53: 40 84 ff                    	testb	%dil, %dil
100004f56: 0f 48 fa                    	cmovsl	%edx, %edi
100004f59: 40 88 7c 30 04              	movb	%dil, 4(%rax,%rsi)
100004f5e: 0f b6 7c 31 05              	movzbl	5(%rcx,%rsi), %edi
100004f63: 40 84 ff                    	testb	%dil, %dil
100004f66: 0f 48 fa                    	cmovsl	%edx, %edi
100004f69: 40 88 7c 30 05              	movb	%dil, 5(%rax,%rsi)
100004f6e: 0f b6 7c 31 06              	movzbl	6(%rcx,%rsi), %edi
100004f73: 40 84 ff                    	testb	%dil, %dil
100004f76: 0f 48 fa                    	cmovsl	%edx, %edi
100004f79: 40 88 7c 30 06              	movb	%dil, 6(%rax,%rsi)
100004f7e: 0f b6 7c 31 07              	movzbl	7(%rcx,%rsi), %edi
100004f83: 40 84 ff                    	testb	%dil, %dil
100004f86: 0f 48 fa                    	cmovsl	%edx, %edi
100004f89: 40 88 7c 30 07              	movb	%dil, 7(%rax,%rsi)
100004f8e: 48 83 c6 08                 	addq	$8, %rsi
100004f92: 81 fe 00 00 01 00           	cmpl	$65536, %esi
100004f98: 0f 85 72 ff ff ff           	jne	-142 <__ZN11LineNetwork7forwardEv+0xe60>
100004f9e: e9 c6 04 00 00              	jmp	1222 <__ZN11LineNetwork7forwardEv+0x13b9>
100004fa3: 31 d2                       	xorl	%edx, %edx
100004fa5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100004faf: 90                          	nop
100004fb0: c5 7a 6f 34 91              	vmovdqu	(%rcx,%rdx,4), %xmm14
100004fb5: c5 7a 6f 7c 91 10           	vmovdqu	16(%rcx,%rdx,4), %xmm15
100004fbb: c5 fa 6f 54 91 20           	vmovdqu	32(%rcx,%rdx,4), %xmm2
100004fc1: c5 fa 6f 5c 91 30           	vmovdqu	48(%rcx,%rdx,4), %xmm3
100004fc7: c5 79 6f 1d a1 21 00 00     	vmovdqa	8609(%rip), %xmm11
100004fcf: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
100004fd4: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
100004fd9: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100004fdd: c5 79 6f 05 9b 21 00 00     	vmovdqa	8603(%rip), %xmm8
100004fe5: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
100004fea: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
100004fef: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100004ff3: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
100004ff9: c5 fa 6f 64 91 70           	vmovdqu	112(%rcx,%rdx,4), %xmm4
100004fff: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100005004: c4 e3 fd 00 6c 91 60 4e     	vpermq	$78, 96(%rcx,%rdx,4), %ymm5
10000500c: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005012: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100005017: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000501b: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100005021: c5 fa 6f 74 91 50           	vmovdqu	80(%rcx,%rdx,4), %xmm6
100005027: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
10000502c: c4 63 fd 00 4c 91 40 4e     	vpermq	$78, 64(%rcx,%rdx,4), %ymm9
100005034: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
10000503a: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
10000503f: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100005044: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000504a: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100005050: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005056: c5 79 6f 05 32 21 00 00     	vmovdqa	8498(%rip), %xmm8
10000505e: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100005063: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100005068: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
10000506c: c5 79 6f 1d 2c 21 00 00     	vmovdqa	8492(%rip), %xmm11
100005074: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
100005079: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000507e: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
100005082: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
100005088: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
10000508d: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
100005092: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005096: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
10000509c: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
1000050a1: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
1000050a6: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
1000050aa: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000050b0: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000050b6: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
1000050bc: c5 79 6f 1d ec 20 00 00     	vmovdqa	8428(%rip), %xmm11
1000050c4: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
1000050c9: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
1000050ce: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
1000050d2: c5 f9 6f 0d e6 20 00 00     	vmovdqa	8422(%rip), %xmm1
1000050da: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
1000050de: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
1000050e3: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
1000050e8: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000050ec: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
1000050f2: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
1000050f7: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
1000050fc: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005100: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005106: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
10000510b: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100005110: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100005114: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000511a: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005120: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100005126: c5 f9 6f 0d a2 20 00 00     	vmovdqa	8354(%rip), %xmm1
10000512e: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100005133: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100005138: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000513c: c5 f9 6f 15 9c 20 00 00     	vmovdqa	8348(%rip), %xmm2
100005144: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100005148: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
10000514d: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100005152: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005156: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
10000515c: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100005161: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100005166: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000516a: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005170: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
100005175: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
10000517a: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
10000517e: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100005184: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
10000518a: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
100005190: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
100005195: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
10000519a: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
10000519f: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
1000051a4: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000051a9: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
1000051ad: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000051b1: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
1000051b5: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
1000051b9: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000051bd: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000051c1: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000051c5: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000051c9: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
1000051cf: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
1000051d5: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
1000051db: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
1000051e1: c5 fe 7f 4c 90 40           	vmovdqu	%ymm1, 64(%rax,%rdx,4)
1000051e7: c5 fe 7f 44 90 60           	vmovdqu	%ymm0, 96(%rax,%rdx,4)
1000051ed: c5 fe 7f 6c 90 20           	vmovdqu	%ymm5, 32(%rax,%rdx,4)
1000051f3: c5 fe 7f 14 90              	vmovdqu	%ymm2, (%rax,%rdx,4)
1000051f8: c5 7a 6f a4 91 80 00 00 00  	vmovdqu	128(%rcx,%rdx,4), %xmm12
100005201: c5 7a 6f ac 91 90 00 00 00  	vmovdqu	144(%rcx,%rdx,4), %xmm13
10000520a: c5 7a 6f b4 91 a0 00 00 00  	vmovdqu	160(%rcx,%rdx,4), %xmm14
100005213: c5 fa 6f 9c 91 b0 00 00 00  	vmovdqu	176(%rcx,%rdx,4), %xmm3
10000521c: c5 f9 6f 05 4c 1f 00 00     	vmovdqa	8012(%rip), %xmm0
100005224: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
100005229: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
10000522e: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100005232: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100005236: c5 f9 6f 05 42 1f 00 00     	vmovdqa	8002(%rip), %xmm0
10000523e: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100005243: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
100005248: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
10000524c: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005250: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
100005256: c5 fa 6f a4 91 f0 00 00 00  	vmovdqu	240(%rcx,%rdx,4), %xmm4
10000525f: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100005264: c4 e3 fd 00 ac 91 e0 00 00 00 4e    	vpermq	$78, 224(%rcx,%rdx,4), %ymm5
10000526f: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005275: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
10000527a: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000527e: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
100005284: c5 fa 6f b4 91 d0 00 00 00  	vmovdqu	208(%rcx,%rdx,4), %xmm6
10000528d: c4 e3 fd 00 bc 91 c0 00 00 00 4e    	vpermq	$78, 192(%rcx,%rdx,4), %ymm7
100005298: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
10000529d: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000052a3: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
1000052a8: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
1000052ac: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000052b2: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
1000052b8: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
1000052be: c5 79 6f 3d ca 1e 00 00     	vmovdqa	7882(%rip), %xmm15
1000052c6: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
1000052cb: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
1000052d0: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
1000052d4: c5 f9 6f 05 c4 1e 00 00     	vmovdqa	7876(%rip), %xmm0
1000052dc: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000052e1: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000052e6: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000052ea: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
1000052f0: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
1000052f5: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
1000052fa: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000052fe: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005304: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005309: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
10000530e: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100005312: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005318: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
10000531e: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005324: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005329: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000532e: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100005332: c5 f9 6f 05 86 1e 00 00     	vmovdqa	7814(%rip), %xmm0
10000533a: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
10000533f: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005344: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005348: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
10000534e: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005353: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100005358: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
10000535c: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005361: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005366: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
10000536a: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005370: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005376: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
10000537c: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
100005382: c5 79 6f 3d 46 1e 00 00     	vmovdqa	7750(%rip), %xmm15
10000538a: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
10000538f: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
100005394: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005398: c5 f9 6f 05 40 1e 00 00     	vmovdqa	7744(%rip), %xmm0
1000053a0: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
1000053a5: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
1000053aa: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000053ae: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
1000053b4: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
1000053b9: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
1000053be: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
1000053c2: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
1000053c7: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
1000053cc: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
1000053d0: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
1000053d6: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
1000053dc: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
1000053e2: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
1000053e8: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
1000053ed: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
1000053f2: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
1000053f7: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000053fc: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100005400: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005404: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005408: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
10000540c: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100005410: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100005414: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005418: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
10000541c: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100005422: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100005428: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
10000542e: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005434: c5 fe 7f 8c 90 c0 00 00 00  	vmovdqu	%ymm1, 192(%rax,%rdx,4)
10000543d: c5 fe 7f 84 90 e0 00 00 00  	vmovdqu	%ymm0, 224(%rax,%rdx,4)
100005446: c5 fe 7f 9c 90 a0 00 00 00  	vmovdqu	%ymm3, 160(%rax,%rdx,4)
10000544f: c5 fe 7f 94 90 80 00 00 00  	vmovdqu	%ymm2, 128(%rax,%rdx,4)
100005458: 48 83 c2 40                 	addq	$64, %rdx
10000545c: 48 81 fa 00 40 00 00        	cmpq	$16384, %rdx
100005463: 0f 85 47 fb ff ff           	jne	-1209 <__ZN11LineNetwork7forwardEv+0xf00>
100005469: 0f b6 43 24                 	movzbl	36(%rbx), %eax
10000546d: 48 83 f0 01                 	xorq	$1, %rax
100005471: 88 43 24                    	movb	%al, 36(%rbx)
100005474: 31 c9                       	xorl	%ecx, %ecx
100005476: 84 c0                       	testb	%al, %al
100005478: 0f 94 c1                    	sete	%cl
10000547b: 4c 8b 64 cb 28              	movq	40(%rbx,%rcx,8), %r12
100005480: 48 8b 44 c3 28              	movq	40(%rbx,%rax,8), %rax
100005485: 48 89 45 88                 	movq	%rax, -120(%rbp)
100005489: 31 c0                       	xorl	%eax, %eax
10000548b: eb 1b                       	jmp	27 <__ZN11LineNetwork7forwardEv+0x13f8>
10000548d: 0f 1f 00                    	nopl	(%rax)
100005490: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100005494: 48 ff c0                    	incq	%rax
100005497: 4c 8b 65 c8                 	movq	-56(%rbp), %r12
10000549b: 49 ff c4                    	incq	%r12
10000549e: 48 83 f8 20                 	cmpq	$32, %rax
1000054a2: 0f 84 28 01 00 00           	je	296 <__ZN11LineNetwork7forwardEv+0x1520>
1000054a8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
1000054ac: 48 8d 04 c0                 	leaq	(%rax,%rax,8), %rax
1000054b0: 48 c1 e0 04                 	shlq	$4, %rax
1000054b4: 48 8d 15 15 26 00 00        	leaq	9749(%rip), %rdx
1000054bb: 48 8d 34 02                 	leaq	(%rdx,%rax), %rsi
1000054bf: 48 89 75 98                 	movq	%rsi, -104(%rbp)
1000054c3: 48 8d 34 10                 	leaq	(%rax,%rdx), %rsi
1000054c7: 48 83 c6 30                 	addq	$48, %rsi
1000054cb: 48 89 75 a0                 	movq	%rsi, -96(%rbp)
1000054cf: 48 8d 44 10 60              	leaq	96(%rax,%rdx), %rax
1000054d4: 48 89 45 a8                 	movq	%rax, -88(%rbp)
1000054d8: 4c 89 65 c8                 	movq	%r12, -56(%rbp)
1000054dc: 48 8b 5d 88                 	movq	-120(%rbp), %rbx
1000054e0: 31 c0                       	xorl	%eax, %eax
1000054e2: eb 2c                       	jmp	44 <__ZN11LineNetwork7forwardEv+0x1460>
1000054e4: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000054ee: 66 90                       	nop
1000054f0: 48 8b 5d b8                 	movq	-72(%rbp), %rbx
1000054f4: 48 81 c3 00 08 00 00        	addq	$2048, %rbx
1000054fb: 4c 8b 65 b0                 	movq	-80(%rbp), %r12
1000054ff: 49 81 c4 00 04 00 00        	addq	$1024, %r12
100005506: 48 8b 45 c0                 	movq	-64(%rbp), %rax
10000550a: 48 83 f8 3d                 	cmpq	$61, %rax
10000550e: 73 80                       	jae	-128 <__ZN11LineNetwork7forwardEv+0x13e0>
100005510: 48 83 c0 02                 	addq	$2, %rax
100005514: 48 89 45 c0                 	movq	%rax, -64(%rbp)
100005518: 4c 89 65 b0                 	movq	%r12, -80(%rbp)
10000551c: 48 89 5d b8                 	movq	%rbx, -72(%rbp)
100005520: 45 31 ed                    	xorl	%r13d, %r13d
100005523: eb 21                       	jmp	33 <__ZN11LineNetwork7forwardEv+0x1496>
100005525: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000552f: 90                          	nop
100005530: 41 88 04 24                 	movb	%al, (%r12)
100005534: 49 83 c5 02                 	addq	$2, %r13
100005538: 48 83 c3 20                 	addq	$32, %rbx
10000553c: 49 83 c4 20                 	addq	$32, %r12
100005540: 49 83 fd 3d                 	cmpq	$61, %r13
100005544: 73 aa                       	jae	-86 <__ZN11LineNetwork7forwardEv+0x1440>
100005546: 48 89 df                    	movq	%rbx, %rdi
100005549: 48 8b 75 98                 	movq	-104(%rbp), %rsi
10000554d: c5 f8 77                    	vzeroupper
100005550: e8 cb 15 00 00              	callq	5579 <__ZN11LineNetwork7forwardEv+0x2a70>
100005555: 41 89 c6                    	movl	%eax, %r14d
100005558: 48 8d bb 00 04 00 00        	leaq	1024(%rbx), %rdi
10000555f: 48 8b 75 a0                 	movq	-96(%rbp), %rsi
100005563: e8 b8 15 00 00              	callq	5560 <__ZN11LineNetwork7forwardEv+0x2a70>
100005568: 41 89 c7                    	movl	%eax, %r15d
10000556b: 45 01 f7                    	addl	%r14d, %r15d
10000556e: 48 8d bb 00 08 00 00        	leaq	2048(%rbx), %rdi
100005575: 48 8b 75 a8                 	movq	-88(%rbp), %rsi
100005579: e8 a2 15 00 00              	callq	5538 <__ZN11LineNetwork7forwardEv+0x2a70>
10000557e: 44 01 f8                    	addl	%r15d, %eax
100005581: 48 8b 4d d0                 	movq	-48(%rbp), %rcx
100005585: 48 8d 15 44 37 00 00        	leaq	14148(%rip), %rdx
10000558c: 0f be 0c 11                 	movsbl	(%rcx,%rdx), %ecx
100005590: 01 c1                       	addl	%eax, %ecx
100005592: c1 e1 04                    	shll	$4, %ecx
100005595: 8d 04 49                    	leal	(%rcx,%rcx,2), %eax
100005598: 48 98                       	cltq
10000559a: 48 69 c8 09 04 02 81        	imulq	$-2130574327, %rax, %rcx
1000055a1: 48 c1 e9 20                 	shrq	$32, %rcx
1000055a5: 01 c8                       	addl	%ecx, %eax
1000055a7: 89 c1                       	movl	%eax, %ecx
1000055a9: c1 e9 1f                    	shrl	$31, %ecx
1000055ac: c1 f8 0d                    	sarl	$13, %eax
1000055af: 01 c8                       	addl	%ecx, %eax
1000055b1: 3d 80 00 00 00              	cmpl	$128, %eax
1000055b6: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x150d>
1000055b8: b8 7f 00 00 00              	movl	$127, %eax
1000055bd: 83 f8 81                    	cmpl	$-127, %eax
1000055c0: 0f 8f 6a ff ff ff           	jg	-150 <__ZN11LineNetwork7forwardEv+0x1480>
1000055c6: b8 81 00 00 00              	movl	$129, %eax
1000055cb: e9 60 ff ff ff              	jmp	-160 <__ZN11LineNetwork7forwardEv+0x1480>
1000055d0: 48 8b 5d 90                 	movq	-112(%rbp), %rbx
1000055d4: 0f b6 4b 24                 	movzbl	36(%rbx), %ecx
1000055d8: 48 83 f1 01                 	xorq	$1, %rcx
1000055dc: 88 4b 24                    	movb	%cl, 36(%rbx)
1000055df: 31 c0                       	xorl	%eax, %eax
1000055e1: 84 c9                       	testb	%cl, %cl
1000055e3: 0f 94 c0                    	sete	%al
1000055e6: 48 8b 44 c3 28              	movq	40(%rbx,%rax,8), %rax
1000055eb: 48 8b 4c cb 28              	movq	40(%rbx,%rcx,8), %rcx
1000055f0: 48 8d 91 00 80 00 00        	leaq	32768(%rcx), %rdx
1000055f7: 48 39 d0                    	cmpq	%rdx, %rax
1000055fa: 0f 83 b3 00 00 00           	jae	179 <__ZN11LineNetwork7forwardEv+0x1603>
100005600: 48 8d 90 00 80 00 00        	leaq	32768(%rax), %rdx
100005607: 48 39 d1                    	cmpq	%rdx, %rcx
10000560a: 0f 83 a3 00 00 00           	jae	163 <__ZN11LineNetwork7forwardEv+0x1603>
100005610: 31 d2                       	xorl	%edx, %edx
100005612: 31 f6                       	xorl	%esi, %esi
100005614: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
10000561e: 66 90                       	nop
100005620: 0f b6 3c 31                 	movzbl	(%rcx,%rsi), %edi
100005624: 40 84 ff                    	testb	%dil, %dil
100005627: 0f 48 fa                    	cmovsl	%edx, %edi
10000562a: 40 88 3c 30                 	movb	%dil, (%rax,%rsi)
10000562e: 0f b6 7c 31 01              	movzbl	1(%rcx,%rsi), %edi
100005633: 40 84 ff                    	testb	%dil, %dil
100005636: 0f 48 fa                    	cmovsl	%edx, %edi
100005639: 40 88 7c 30 01              	movb	%dil, 1(%rax,%rsi)
10000563e: 0f b6 7c 31 02              	movzbl	2(%rcx,%rsi), %edi
100005643: 40 84 ff                    	testb	%dil, %dil
100005646: 0f 48 fa                    	cmovsl	%edx, %edi
100005649: 40 88 7c 30 02              	movb	%dil, 2(%rax,%rsi)
10000564e: 0f b6 7c 31 03              	movzbl	3(%rcx,%rsi), %edi
100005653: 40 84 ff                    	testb	%dil, %dil
100005656: 0f 48 fa                    	cmovsl	%edx, %edi
100005659: 40 88 7c 30 03              	movb	%dil, 3(%rax,%rsi)
10000565e: 0f b6 7c 31 04              	movzbl	4(%rcx,%rsi), %edi
100005663: 40 84 ff                    	testb	%dil, %dil
100005666: 0f 48 fa                    	cmovsl	%edx, %edi
100005669: 40 88 7c 30 04              	movb	%dil, 4(%rax,%rsi)
10000566e: 0f b6 7c 31 05              	movzbl	5(%rcx,%rsi), %edi
100005673: 40 84 ff                    	testb	%dil, %dil
100005676: 0f 48 fa                    	cmovsl	%edx, %edi
100005679: 40 88 7c 30 05              	movb	%dil, 5(%rax,%rsi)
10000567e: 0f b6 7c 31 06              	movzbl	6(%rcx,%rsi), %edi
100005683: 40 84 ff                    	testb	%dil, %dil
100005686: 0f 48 fa                    	cmovsl	%edx, %edi
100005689: 40 88 7c 30 06              	movb	%dil, 6(%rax,%rsi)
10000568e: 0f b6 7c 31 07              	movzbl	7(%rcx,%rsi), %edi
100005693: 40 84 ff                    	testb	%dil, %dil
100005696: 0f 48 fa                    	cmovsl	%edx, %edi
100005699: 40 88 7c 30 07              	movb	%dil, 7(%rax,%rsi)
10000569e: 48 83 c6 08                 	addq	$8, %rsi
1000056a2: 81 fe 00 80 00 00           	cmpl	$32768, %esi
1000056a8: 0f 85 72 ff ff ff           	jne	-142 <__ZN11LineNetwork7forwardEv+0x1570>
1000056ae: e9 c6 04 00 00              	jmp	1222 <__ZN11LineNetwork7forwardEv+0x1ac9>
1000056b3: 31 d2                       	xorl	%edx, %edx
1000056b5: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
1000056bf: 90                          	nop
1000056c0: c5 7a 6f 34 91              	vmovdqu	(%rcx,%rdx,4), %xmm14
1000056c5: c5 7a 6f 7c 91 10           	vmovdqu	16(%rcx,%rdx,4), %xmm15
1000056cb: c5 fa 6f 54 91 20           	vmovdqu	32(%rcx,%rdx,4), %xmm2
1000056d1: c5 fa 6f 5c 91 30           	vmovdqu	48(%rcx,%rdx,4), %xmm3
1000056d7: c5 79 6f 1d 91 1a 00 00     	vmovdqa	6801(%rip), %xmm11
1000056df: c4 c2 61 00 e3              	vpshufb	%xmm11, %xmm3, %xmm4
1000056e4: c4 c2 69 00 eb              	vpshufb	%xmm11, %xmm2, %xmm5
1000056e9: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
1000056ed: c5 79 6f 05 8b 1a 00 00     	vmovdqa	6795(%rip), %xmm8
1000056f5: c4 c2 01 00 e8              	vpshufb	%xmm8, %xmm15, %xmm5
1000056fa: c4 c2 09 00 f0              	vpshufb	%xmm8, %xmm14, %xmm6
1000056ff: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005703: c4 63 51 02 d4 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm10
100005709: c5 fa 6f 64 91 70           	vmovdqu	112(%rcx,%rdx,4), %xmm4
10000570f: c4 c2 59 00 f3              	vpshufb	%xmm11, %xmm4, %xmm6
100005714: c4 e3 fd 00 6c 91 60 4e     	vpermq	$78, 96(%rcx,%rdx,4), %ymm5
10000571c: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005722: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
100005727: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000572b: c4 e3 7d 38 fe 01           	vinserti128	$1, %xmm6, %ymm0, %ymm7
100005731: c5 fa 6f 74 91 50           	vmovdqu	80(%rcx,%rdx,4), %xmm6
100005737: c4 42 49 00 e0              	vpshufb	%xmm8, %xmm6, %xmm12
10000573c: c4 63 fd 00 4c 91 40 4e     	vpermq	$78, 64(%rcx,%rdx,4), %ymm9
100005744: c4 43 7d 39 c9 01           	vextracti128	$1, %ymm9, %xmm9
10000574a: c4 42 31 00 e8              	vpshufb	%xmm8, %xmm9, %xmm13
10000574f: c4 c1 11 62 c4              	vpunpckldq	%xmm12, %xmm13, %xmm0
100005754: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000575a: c4 e3 7d 02 c7 c0           	vpblendd	$192, %ymm7, %ymm0, %ymm0
100005760: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005766: c5 79 6f 05 22 1a 00 00     	vmovdqa	6690(%rip), %xmm8
10000576e: c4 c2 61 00 c0              	vpshufb	%xmm8, %xmm3, %xmm0
100005773: c4 c2 69 00 f8              	vpshufb	%xmm8, %xmm2, %xmm7
100005778: c5 c1 62 c0                 	vpunpckldq	%xmm0, %xmm7, %xmm0
10000577c: c5 79 6f 1d 1c 1a 00 00     	vmovdqa	6684(%rip), %xmm11
100005784: c4 c2 01 00 fb              	vpshufb	%xmm11, %xmm15, %xmm7
100005789: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
10000578e: c5 f1 62 cf                 	vpunpckldq	%xmm7, %xmm1, %xmm1
100005792: c4 63 71 02 e0 0c           	vpblendd	$12, %xmm0, %xmm1, %xmm12
100005798: c4 c2 59 00 c8              	vpshufb	%xmm8, %xmm4, %xmm1
10000579d: c4 c2 51 00 f8              	vpshufb	%xmm8, %xmm5, %xmm7
1000057a2: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000057a6: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
1000057ac: c4 c2 49 00 fb              	vpshufb	%xmm11, %xmm6, %xmm7
1000057b1: c4 c2 31 00 c3              	vpshufb	%xmm11, %xmm9, %xmm0
1000057b6: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
1000057ba: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000057c0: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
1000057c6: c4 63 1d 02 e0 f0           	vpblendd	$240, %ymm0, %ymm12, %ymm12
1000057cc: c5 79 6f 1d dc 19 00 00     	vmovdqa	6620(%rip), %xmm11
1000057d4: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
1000057d9: c4 c2 69 00 cb              	vpshufb	%xmm11, %xmm2, %xmm1
1000057de: c5 71 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm8
1000057e2: c5 f9 6f 0d d6 19 00 00     	vmovdqa	6614(%rip), %xmm1
1000057ea: c5 f9 6f c1                 	vmovdqa	%xmm1, %xmm0
1000057ee: c4 e2 01 00 c9              	vpshufb	%xmm1, %xmm15, %xmm1
1000057f3: c4 e2 09 00 f8              	vpshufb	%xmm0, %xmm14, %xmm7
1000057f8: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
1000057fc: c4 43 71 02 e8 0c           	vpblendd	$12, %xmm8, %xmm1, %xmm13
100005802: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005807: c4 c2 51 00 fb              	vpshufb	%xmm11, %xmm5, %xmm7
10000580c: c5 c1 62 c9                 	vpunpckldq	%xmm1, %xmm7, %xmm1
100005810: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005816: c4 e2 49 00 f8              	vpshufb	%xmm0, %xmm6, %xmm7
10000581b: c4 e2 31 00 c0              	vpshufb	%xmm0, %xmm9, %xmm0
100005820: c5 f9 62 c7                 	vpunpckldq	%xmm7, %xmm0, %xmm0
100005824: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
10000582a: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005830: c4 63 15 02 e8 f0           	vpblendd	$240, %ymm0, %ymm13, %ymm13
100005836: c5 f9 6f 0d 92 19 00 00     	vmovdqa	6546(%rip), %xmm1
10000583e: c4 e2 61 00 d9              	vpshufb	%xmm1, %xmm3, %xmm3
100005843: c4 e2 69 00 d1              	vpshufb	%xmm1, %xmm2, %xmm2
100005848: c5 e9 62 c3                 	vpunpckldq	%xmm3, %xmm2, %xmm0
10000584c: c5 f9 6f 15 8c 19 00 00     	vmovdqa	6540(%rip), %xmm2
100005854: c5 f9 6f fa                 	vmovdqa	%xmm2, %xmm7
100005858: c4 e2 01 00 d2              	vpshufb	%xmm2, %xmm15, %xmm2
10000585d: c4 e2 09 00 df              	vpshufb	%xmm7, %xmm14, %xmm3
100005862: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005866: c4 e3 69 02 c0 0c           	vpblendd	$12, %xmm0, %xmm2, %xmm0
10000586c: c4 e2 59 00 d1              	vpshufb	%xmm1, %xmm4, %xmm2
100005871: c4 e2 51 00 d9              	vpshufb	%xmm1, %xmm5, %xmm3
100005876: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
10000587a: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005880: c4 e2 49 00 df              	vpshufb	%xmm7, %xmm6, %xmm3
100005885: c4 e2 31 00 e7              	vpshufb	%xmm7, %xmm9, %xmm4
10000588a: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
10000588e: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100005894: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
10000589a: c4 e3 7d 02 ca f0           	vpblendd	$240, %ymm2, %ymm0, %ymm1
1000058a0: c4 41 39 ef c0              	vpxor	%xmm8, %xmm8, %xmm8
1000058a5: c4 c2 2d 3c d0              	vpmaxsb	%ymm8, %ymm10, %ymm2
1000058aa: c4 c2 1d 3c d8              	vpmaxsb	%ymm8, %ymm12, %ymm3
1000058af: c4 c2 15 3c c0              	vpmaxsb	%ymm8, %ymm13, %ymm0
1000058b4: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
1000058b9: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
1000058bd: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
1000058c1: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
1000058c5: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
1000058c9: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
1000058cd: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
1000058d1: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
1000058d5: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
1000058d9: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
1000058df: c4 e3 5d 38 e8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm5
1000058e5: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
1000058eb: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
1000058f1: c5 fe 7f 4c 90 40           	vmovdqu	%ymm1, 64(%rax,%rdx,4)
1000058f7: c5 fe 7f 44 90 60           	vmovdqu	%ymm0, 96(%rax,%rdx,4)
1000058fd: c5 fe 7f 6c 90 20           	vmovdqu	%ymm5, 32(%rax,%rdx,4)
100005903: c5 fe 7f 14 90              	vmovdqu	%ymm2, (%rax,%rdx,4)
100005908: c5 7a 6f a4 91 80 00 00 00  	vmovdqu	128(%rcx,%rdx,4), %xmm12
100005911: c5 7a 6f ac 91 90 00 00 00  	vmovdqu	144(%rcx,%rdx,4), %xmm13
10000591a: c5 7a 6f b4 91 a0 00 00 00  	vmovdqu	160(%rcx,%rdx,4), %xmm14
100005923: c5 fa 6f 9c 91 b0 00 00 00  	vmovdqu	176(%rcx,%rdx,4), %xmm3
10000592c: c5 f9 6f 05 3c 18 00 00     	vmovdqa	6204(%rip), %xmm0
100005934: c4 e2 61 00 e0              	vpshufb	%xmm0, %xmm3, %xmm4
100005939: c4 e2 09 00 e8              	vpshufb	%xmm0, %xmm14, %xmm5
10000593e: c5 f9 6f d0                 	vmovdqa	%xmm0, %xmm2
100005942: c5 d1 62 e4                 	vpunpckldq	%xmm4, %xmm5, %xmm4
100005946: c5 f9 6f 05 32 18 00 00     	vmovdqa	6194(%rip), %xmm0
10000594e: c4 e2 11 00 e8              	vpshufb	%xmm0, %xmm13, %xmm5
100005953: c4 e2 19 00 f0              	vpshufb	%xmm0, %xmm12, %xmm6
100005958: c5 f9 6f c8                 	vmovdqa	%xmm0, %xmm1
10000595c: c5 c9 62 ed                 	vpunpckldq	%xmm5, %xmm6, %xmm5
100005960: c4 63 51 02 cc 0c           	vpblendd	$12, %xmm4, %xmm5, %xmm9
100005966: c5 fa 6f a4 91 f0 00 00 00  	vmovdqu	240(%rcx,%rdx,4), %xmm4
10000596f: c4 e2 59 00 f2              	vpshufb	%xmm2, %xmm4, %xmm6
100005974: c4 e3 fd 00 ac 91 e0 00 00 00 4e    	vpermq	$78, 224(%rcx,%rdx,4), %ymm5
10000597f: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100005985: c4 e2 51 00 fa              	vpshufb	%xmm2, %xmm5, %xmm7
10000598a: c5 c1 62 f6                 	vpunpckldq	%xmm6, %xmm7, %xmm6
10000598e: c4 63 7d 38 d6 01           	vinserti128	$1, %xmm6, %ymm0, %ymm10
100005994: c5 fa 6f b4 91 d0 00 00 00  	vmovdqu	208(%rcx,%rdx,4), %xmm6
10000599d: c4 e3 fd 00 bc 91 c0 00 00 00 4e    	vpermq	$78, 192(%rcx,%rdx,4), %ymm7
1000059a8: c4 e2 49 00 c0              	vpshufb	%xmm0, %xmm6, %xmm0
1000059ad: c4 e3 7d 39 ff 01           	vextracti128	$1, %ymm7, %xmm7
1000059b3: c4 e2 41 00 c9              	vpshufb	%xmm1, %xmm7, %xmm1
1000059b8: c5 f1 62 c0                 	vpunpckldq	%xmm0, %xmm1, %xmm0
1000059bc: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
1000059c2: c4 c3 7d 02 c2 c0           	vpblendd	$192, %ymm10, %ymm0, %ymm0
1000059c8: c4 63 35 02 c8 f0           	vpblendd	$240, %ymm0, %ymm9, %ymm9
1000059ce: c5 79 6f 3d ba 17 00 00     	vmovdqa	6074(%rip), %xmm15
1000059d6: c4 c2 61 00 c7              	vpshufb	%xmm15, %xmm3, %xmm0
1000059db: c4 c2 09 00 cf              	vpshufb	%xmm15, %xmm14, %xmm1
1000059e0: c5 71 62 d0                 	vpunpckldq	%xmm0, %xmm1, %xmm10
1000059e4: c5 f9 6f 05 b4 17 00 00     	vmovdqa	6068(%rip), %xmm0
1000059ec: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
1000059f1: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
1000059f6: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
1000059fa: c4 43 71 02 d2 0c           	vpblendd	$12, %xmm10, %xmm1, %xmm10
100005a00: c4 c2 59 00 cf              	vpshufb	%xmm15, %xmm4, %xmm1
100005a05: c4 c2 51 00 d7              	vpshufb	%xmm15, %xmm5, %xmm2
100005a0a: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005a0e: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005a14: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005a19: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005a1e: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100005a22: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005a28: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005a2e: c4 63 2d 02 d0 f0           	vpblendd	$240, %ymm0, %ymm10, %ymm10
100005a34: c4 c2 61 00 c3              	vpshufb	%xmm11, %xmm3, %xmm0
100005a39: c4 c2 09 00 cb              	vpshufb	%xmm11, %xmm14, %xmm1
100005a3e: c5 71 62 f8                 	vpunpckldq	%xmm0, %xmm1, %xmm15
100005a42: c5 f9 6f 05 76 17 00 00     	vmovdqa	6006(%rip), %xmm0
100005a4a: c4 e2 11 00 c8              	vpshufb	%xmm0, %xmm13, %xmm1
100005a4f: c4 e2 19 00 d0              	vpshufb	%xmm0, %xmm12, %xmm2
100005a54: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005a58: c4 43 71 02 ff 0c           	vpblendd	$12, %xmm15, %xmm1, %xmm15
100005a5e: c4 c2 59 00 cb              	vpshufb	%xmm11, %xmm4, %xmm1
100005a63: c4 c2 51 00 d3              	vpshufb	%xmm11, %xmm5, %xmm2
100005a68: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005a6c: c4 e2 49 00 d0              	vpshufb	%xmm0, %xmm6, %xmm2
100005a71: c4 e2 41 00 c0              	vpshufb	%xmm0, %xmm7, %xmm0
100005a76: c5 f9 62 c2                 	vpunpckldq	%xmm2, %xmm0, %xmm0
100005a7a: c4 e3 7d 38 c9 01           	vinserti128	$1, %xmm1, %ymm0, %ymm1
100005a80: c4 e3 7d 38 c0 01           	vinserti128	$1, %xmm0, %ymm0, %ymm0
100005a86: c4 e3 7d 02 c1 c0           	vpblendd	$192, %ymm1, %ymm0, %ymm0
100005a8c: c4 63 05 02 d8 f0           	vpblendd	$240, %ymm0, %ymm15, %ymm11
100005a92: c5 79 6f 3d 36 17 00 00     	vmovdqa	5942(%rip), %xmm15
100005a9a: c4 c2 61 00 cf              	vpshufb	%xmm15, %xmm3, %xmm1
100005a9f: c4 c2 09 00 d7              	vpshufb	%xmm15, %xmm14, %xmm2
100005aa4: c5 e9 62 c9                 	vpunpckldq	%xmm1, %xmm2, %xmm1
100005aa8: c5 f9 6f 05 30 17 00 00     	vmovdqa	5936(%rip), %xmm0
100005ab0: c4 e2 11 00 d0              	vpshufb	%xmm0, %xmm13, %xmm2
100005ab5: c4 e2 19 00 d8              	vpshufb	%xmm0, %xmm12, %xmm3
100005aba: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005abe: c4 e3 69 02 c9 0c           	vpblendd	$12, %xmm1, %xmm2, %xmm1
100005ac4: c4 c2 59 00 d7              	vpshufb	%xmm15, %xmm4, %xmm2
100005ac9: c4 c2 51 00 df              	vpshufb	%xmm15, %xmm5, %xmm3
100005ace: c5 e1 62 d2                 	vpunpckldq	%xmm2, %xmm3, %xmm2
100005ad2: c4 e2 49 00 d8              	vpshufb	%xmm0, %xmm6, %xmm3
100005ad7: c4 e2 41 00 e0              	vpshufb	%xmm0, %xmm7, %xmm4
100005adc: c5 d9 62 db                 	vpunpckldq	%xmm3, %xmm4, %xmm3
100005ae0: c4 e3 7d 38 d2 01           	vinserti128	$1, %xmm2, %ymm0, %ymm2
100005ae6: c4 e3 7d 38 db 01           	vinserti128	$1, %xmm3, %ymm0, %ymm3
100005aec: c4 e3 65 02 d2 c0           	vpblendd	$192, %ymm2, %ymm3, %ymm2
100005af2: c4 e3 75 02 ca f0           	vpblendd	$240, %ymm2, %ymm1, %ymm1
100005af8: c4 c2 35 3c d0              	vpmaxsb	%ymm8, %ymm9, %ymm2
100005afd: c4 c2 2d 3c d8              	vpmaxsb	%ymm8, %ymm10, %ymm3
100005b02: c4 c2 25 3c c0              	vpmaxsb	%ymm8, %ymm11, %ymm0
100005b07: c4 c2 75 3c c8              	vpmaxsb	%ymm8, %ymm1, %ymm1
100005b0c: c5 ed 60 e3                 	vpunpcklbw	%ymm3, %ymm2, %ymm4
100005b10: c5 ed 68 d3                 	vpunpckhbw	%ymm3, %ymm2, %ymm2
100005b14: c5 fd 60 d9                 	vpunpcklbw	%ymm1, %ymm0, %ymm3
100005b18: c5 fd 68 c1                 	vpunpckhbw	%ymm1, %ymm0, %ymm0
100005b1c: c5 dd 61 cb                 	vpunpcklwd	%ymm3, %ymm4, %ymm1
100005b20: c5 dd 69 db                 	vpunpckhwd	%ymm3, %ymm4, %ymm3
100005b24: c5 ed 61 e0                 	vpunpcklwd	%ymm0, %ymm2, %ymm4
100005b28: c5 ed 69 c0                 	vpunpckhwd	%ymm0, %ymm2, %ymm0
100005b2c: c4 e3 75 38 d3 01           	vinserti128	$1, %xmm3, %ymm1, %ymm2
100005b32: c4 e3 75 46 cb 31           	vperm2i128	$49, %ymm3, %ymm1, %ymm1
100005b38: c4 e3 5d 38 d8 01           	vinserti128	$1, %xmm0, %ymm4, %ymm3
100005b3e: c4 e3 5d 46 c0 31           	vperm2i128	$49, %ymm0, %ymm4, %ymm0
100005b44: c5 fe 7f 8c 90 c0 00 00 00  	vmovdqu	%ymm1, 192(%rax,%rdx,4)
100005b4d: c5 fe 7f 84 90 e0 00 00 00  	vmovdqu	%ymm0, 224(%rax,%rdx,4)
100005b56: c5 fe 7f 9c 90 a0 00 00 00  	vmovdqu	%ymm3, 160(%rax,%rdx,4)
100005b5f: c5 fe 7f 94 90 80 00 00 00  	vmovdqu	%ymm2, 128(%rax,%rdx,4)
100005b68: 48 83 c2 40                 	addq	$64, %rdx
100005b6c: 48 81 fa 00 20 00 00        	cmpq	$8192, %rdx
100005b73: 0f 85 47 fb ff ff           	jne	-1209 <__ZN11LineNetwork7forwardEv+0x1610>
100005b79: 0f b6 43 24                 	movzbl	36(%rbx), %eax
100005b7d: 48 83 f0 01                 	xorq	$1, %rax
100005b81: 88 43 24                    	movb	%al, 36(%rbx)
100005b84: 31 c9                       	xorl	%ecx, %ecx
100005b86: 84 c0                       	testb	%al, %al
100005b88: 0f 94 c1                    	sete	%cl
100005b8b: 4c 8b 7c cb 28              	movq	40(%rbx,%rcx,8), %r15
100005b90: 4c 8b 64 c3 28              	movq	40(%rbx,%rax,8), %r12
100005b95: 31 c0                       	xorl	%eax, %eax
100005b97: 4c 8d 35 52 31 00 00        	leaq	12626(%rip), %r14
100005b9e: eb 18                       	jmp	24 <__ZN11LineNetwork7forwardEv+0x1b08>
100005ba0: 48 8b 45 d0                 	movq	-48(%rbp), %rax
100005ba4: 48 ff c0                    	incq	%rax
100005ba7: 49 83 c7 20                 	addq	$32, %r15
100005bab: 49 81 c4 00 04 00 00        	addq	$1024, %r12
100005bb2: 48 83 f8 20                 	cmpq	$32, %rax
100005bb6: 74 75                       	je	117 <__ZN11LineNetwork7forwardEv+0x1b7d>
100005bb8: 48 89 45 d0                 	movq	%rax, -48(%rbp)
100005bbc: 4c 89 e3                    	movq	%r12, %rbx
100005bbf: 45 31 ed                    	xorl	%r13d, %r13d
100005bc2: eb 1d                       	jmp	29 <__ZN11LineNetwork7forwardEv+0x1b31>
100005bc4: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005bce: 66 90                       	nop
100005bd0: 43 88 04 2f                 	movb	%al, (%r15,%r13)
100005bd4: 49 ff c5                    	incq	%r13
100005bd7: 48 83 c3 20                 	addq	$32, %rbx
100005bdb: 49 83 fd 20                 	cmpq	$32, %r13
100005bdf: 74 bf                       	je	-65 <__ZN11LineNetwork7forwardEv+0x1af0>
100005be1: 48 89 df                    	movq	%rbx, %rdi
100005be4: 4c 89 f6                    	movq	%r14, %rsi
100005be7: c5 f8 77                    	vzeroupper
100005bea: e8 b1 11 00 00              	callq	4529 <__ZN11LineNetwork7forwardEv+0x2cf0>
100005bef: c1 e0 05                    	shll	$5, %eax
100005bf2: 89 c1                       	movl	%eax, %ecx
100005bf4: 83 c1 20                    	addl	$32, %ecx
100005bf7: 48 63 c9                    	movslq	%ecx, %rcx
100005bfa: 48 69 c9 09 04 02 81        	imulq	$-2130574327, %rcx, %rcx
100005c01: 48 c1 e9 20                 	shrq	$32, %rcx
100005c05: 8d 04 01                    	leal	(%rcx,%rax), %eax
100005c08: 83 c0 20                    	addl	$32, %eax
100005c0b: 89 c1                       	movl	%eax, %ecx
100005c0d: c1 e9 1f                    	shrl	$31, %ecx
100005c10: c1 f8 0d                    	sarl	$13, %eax
100005c13: 01 c8                       	addl	%ecx, %eax
100005c15: 3d 80 00 00 00              	cmpl	$128, %eax
100005c1a: 7c 05                       	jl	5 <__ZN11LineNetwork7forwardEv+0x1b71>
100005c1c: b8 7f 00 00 00              	movl	$127, %eax
100005c21: 83 f8 81                    	cmpl	$-127, %eax
100005c24: 7f aa                       	jg	-86 <__ZN11LineNetwork7forwardEv+0x1b20>
100005c26: b8 81 00 00 00              	movl	$129, %eax
100005c2b: eb a3                       	jmp	-93 <__ZN11LineNetwork7forwardEv+0x1b20>
100005c2d: 48 83 c4 58                 	addq	$88, %rsp
100005c31: 5b                          	popq	%rbx
100005c32: 41 5c                       	popq	%r12
100005c34: 41 5d                       	popq	%r13
100005c36: 41 5e                       	popq	%r14
100005c38: 41 5f                       	popq	%r15
100005c3a: 5d                          	popq	%rbp
100005c3b: c3                          	retq
100005c3c: 0f 1f 40 00                 	nopl	(%rax)
100005c40: 55                          	pushq	%rbp
100005c41: 48 89 e5                    	movq	%rsp, %rbp
100005c44: 41 57                       	pushq	%r15
100005c46: 41 56                       	pushq	%r14
100005c48: 41 55                       	pushq	%r13
100005c4a: 41 54                       	pushq	%r12
100005c4c: 53                          	pushq	%rbx
100005c4d: 48 83 e4 e0                 	andq	$-32, %rsp
100005c51: 48 81 ec 20 03 00 00        	subq	$800, %rsp
100005c58: 48 89 8c 24 90 00 00 00     	movq	%rcx, 144(%rsp)
100005c60: 48 89 94 24 88 00 00 00     	movq	%rdx, 136(%rsp)
100005c68: 49 89 fc                    	movq	%rdi, %r12
100005c6b: c4 c1 79 6e c0              	vmovd	%r8d, %xmm0
100005c70: c4 e2 7d 58 c8              	vpbroadcastd	%xmm0, %ymm1
100005c75: 48 8d 86 01 04 00 00        	leaq	1025(%rsi), %rax
100005c7c: 48 89 84 24 80 00 00 00     	movq	%rax, 128(%rsp)
100005c84: 48 89 b4 24 a0 00 00 00     	movq	%rsi, 160(%rsp)
100005c8c: 48 8d 86 02 04 00 00        	leaq	1026(%rsi), %rax
100005c93: 48 89 44 24 78              	movq	%rax, 120(%rsp)
100005c98: 31 db                       	xorl	%ebx, %ebx
100005c9a: 41 bf 7f 00 00 00           	movl	$127, %r15d
100005ca0: 41 bd 81 00 00 00           	movl	$129, %r13d
100005ca6: 48 89 7c 24 28              	movq	%rdi, 40(%rsp)
100005cab: 44 89 44 24 24              	movl	%r8d, 36(%rsp)
100005cb0: c5 fd 7f 8c 24 80 02 00 00  	vmovdqa	%ymm1, 640(%rsp)
100005cb9: c5 7d 6f 3d 5f 16 00 00     	vmovdqa	5727(%rip), %ymm15
100005cc1: c5 7d 6f 35 77 16 00 00     	vmovdqa	5751(%rip), %ymm14
100005cc9: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005cd0: 48 8d 83 f1 07 00 00        	leaq	2033(%rbx), %rax
100005cd7: 48 89 84 24 c0 00 00 00     	movq	%rax, 192(%rsp)
100005cdf: 48 8d 04 db                 	leaq	(%rbx,%rbx,8), %rax
100005ce3: 48 8b 94 24 88 00 00 00     	movq	136(%rsp), %rdx
100005ceb: 48 8d 0c 02                 	leaq	(%rdx,%rax), %rcx
100005cef: 48 83 c1 09                 	addq	$9, %rcx
100005cf3: 48 89 8c 24 b8 00 00 00     	movq	%rcx, 184(%rsp)
100005cfb: 48 8b 8c 24 90 00 00 00     	movq	144(%rsp), %rcx
100005d03: 48 8d 7c 19 01              	leaq	1(%rcx,%rbx), %rdi
100005d08: 48 89 bc 24 b0 00 00 00     	movq	%rdi, 176(%rsp)
100005d10: 48 8d 3c 02                 	leaq	(%rdx,%rax), %rdi
100005d14: 48 89 7c 24 38              	movq	%rdi, 56(%rsp)
100005d19: 4c 8d 14 19                 	leaq	(%rcx,%rbx), %r10
100005d1d: 48 8d 44 02 08              	leaq	8(%rdx,%rax), %rax
100005d22: 48 89 84 24 a8 00 00 00     	movq	%rax, 168(%rsp)
100005d2a: 48 89 de                    	movq	%rbx, %rsi
100005d2d: c4 e1 f9 6e c3              	vmovq	%rbx, %xmm0
100005d32: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005d37: 41 bb 00 00 00 00           	movl	$0, %r11d
100005d3d: 4c 8b 74 24 78              	movq	120(%rsp), %r14
100005d42: 48 8b 9c 24 80 00 00 00     	movq	128(%rsp), %rbx
100005d4a: 31 c9                       	xorl	%ecx, %ecx
100005d4c: 31 c0                       	xorl	%eax, %eax
100005d4e: 48 89 44 24 18              	movq	%rax, 24(%rsp)
100005d53: 48 89 b4 24 98 00 00 00     	movq	%rsi, 152(%rsp)
100005d5b: 4c 89 54 24 10              	movq	%r10, 16(%rsp)
100005d60: c5 fd 7f 84 24 a0 02 00 00  	vmovdqa	%ymm0, 672(%rsp)
100005d69: 0f 1f 80 00 00 00 00        	nopl	(%rax)
100005d70: 4c 89 b4 24 d0 00 00 00     	movq	%r14, 208(%rsp)
100005d78: 4c 89 9c 24 d8 00 00 00     	movq	%r11, 216(%rsp)
100005d80: 48 89 ca                    	movq	%rcx, %rdx
100005d83: 48 c1 e2 0b                 	shlq	$11, %rdx
100005d87: 48 8d 04 16                 	leaq	(%rsi,%rdx), %rax
100005d8b: 4c 01 e0                    	addq	%r12, %rax
100005d8e: 48 03 94 24 c0 00 00 00     	addq	192(%rsp), %rdx
100005d96: 4c 01 e2                    	addq	%r12, %rdx
100005d99: 48 89 8c 24 c8 00 00 00     	movq	%rcx, 200(%rsp)
100005da1: 48 c1 e1 0a                 	shlq	$10, %rcx
100005da5: 48 8b bc 24 a0 00 00 00     	movq	160(%rsp), %rdi
100005dad: 4c 8d 0c 0f                 	leaq	(%rdi,%rcx), %r9
100005db1: 49 81 c1 ff 05 00 00        	addq	$1535, %r9
100005db8: 48 01 f9                    	addq	%rdi, %rcx
100005dbb: 4c 39 c8                    	cmpq	%r9, %rax
100005dbe: 41 0f 92 c3                 	setb	%r11b
100005dc2: 48 39 d1                    	cmpq	%rdx, %rcx
100005dc5: 41 0f 92 c1                 	setb	%r9b
100005dc9: 48 3b 84 24 b8 00 00 00     	cmpq	184(%rsp), %rax
100005dd1: 0f 92 c1                    	setb	%cl
100005dd4: 48 39 94 24 a8 00 00 00     	cmpq	%rdx, 168(%rsp)
100005ddc: 40 0f 92 c7                 	setb	%dil
100005de0: 48 3b 84 24 b0 00 00 00     	cmpq	176(%rsp), %rax
100005de8: 0f 92 c0                    	setb	%al
100005deb: 49 39 d2                    	cmpq	%rdx, %r10
100005dee: 41 0f 92 c2                 	setb	%r10b
100005df2: 45 84 cb                    	testb	%r9b, %r11b
100005df5: 48 89 5c 24 30              	movq	%rbx, 48(%rsp)
100005dfa: 0f 85 e0 0a 00 00           	jne	2784 <__ZN11LineNetwork7forwardEv+0x2830>
100005e00: 40 20 f9                    	andb	%dil, %cl
100005e03: 0f 85 d7 0a 00 00           	jne	2775 <__ZN11LineNetwork7forwardEv+0x2830>
100005e09: b9 00 00 00 00              	movl	$0, %ecx
100005e0e: 44 20 d0                    	andb	%r10b, %al
100005e11: 4c 8b 54 24 10              	movq	16(%rsp), %r10
100005e16: 0f 85 cb 0a 00 00           	jne	2763 <__ZN11LineNetwork7forwardEv+0x2837>
100005e1c: 48 8b 44 24 18              	movq	24(%rsp), %rax
100005e21: 48 c1 e0 07                 	shlq	$7, %rax
100005e25: c4 e1 f9 6e c0              	vmovq	%rax, %xmm0
100005e2a: c4 e2 7d 59 c0              	vpbroadcastq	%xmm0, %ymm0
100005e2f: c5 fd 7f 84 24 c0 02 00 00  	vmovdqa	%ymm0, 704(%rsp)
100005e38: 45 31 f6                    	xorl	%r14d, %r14d
100005e3b: c5 fc 28 05 bd 14 00 00     	vmovaps	5309(%rip), %ymm0
100005e43: c5 fc 29 84 24 60 02 00 00  	vmovaps	%ymm0, 608(%rsp)
100005e4c: c5 fc 28 05 8c 14 00 00     	vmovaps	5260(%rip), %ymm0
100005e54: c5 fc 29 84 24 40 02 00 00  	vmovaps	%ymm0, 576(%rsp)
100005e5d: c5 fc 28 05 5b 14 00 00     	vmovaps	5211(%rip), %ymm0
100005e65: c5 fc 29 84 24 20 02 00 00  	vmovaps	%ymm0, 544(%rsp)
100005e6e: c5 fc 28 05 2a 14 00 00     	vmovaps	5162(%rip), %ymm0
100005e76: c5 fc 29 84 24 00 02 00 00  	vmovaps	%ymm0, 512(%rsp)
100005e7f: c5 fc 28 05 f9 13 00 00     	vmovaps	5113(%rip), %ymm0
100005e87: c5 fc 29 84 24 e0 01 00 00  	vmovaps	%ymm0, 480(%rsp)
100005e90: c5 fc 28 05 c8 13 00 00     	vmovaps	5064(%rip), %ymm0
100005e98: c5 fc 29 84 24 c0 01 00 00  	vmovaps	%ymm0, 448(%rsp)
100005ea1: c5 fc 28 05 97 13 00 00     	vmovaps	5015(%rip), %ymm0
100005ea9: c5 fc 29 84 24 a0 01 00 00  	vmovaps	%ymm0, 416(%rsp)
100005eb2: c5 fc 28 05 66 13 00 00     	vmovaps	4966(%rip), %ymm0
100005eba: c5 fc 29 84 24 80 01 00 00  	vmovaps	%ymm0, 384(%rsp)
100005ec3: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
100005ecd: 0f 1f 00                    	nopl	(%rax)
100005ed0: 48 8b 4c 24 30              	movq	48(%rsp), %rcx
100005ed5: c4 a1 7e 6f 84 71 1f fc ff ff       	vmovdqu	-993(%rcx,%r14,2), %ymm0
100005edf: c4 c2 7d 00 c7              	vpshufb	%ymm15, %ymm0, %ymm0
100005ee4: c4 a1 7e 6f 8c 71 ff fb ff ff       	vmovdqu	-1025(%rcx,%r14,2), %ymm1
100005eee: c4 a1 7e 6f 94 71 00 fc ff ff       	vmovdqu	-1024(%rcx,%r14,2), %ymm2
100005ef8: c4 c2 75 00 ce              	vpshufb	%ymm14, %ymm1, %ymm1
100005efd: c4 e3 75 02 c0 cc           	vpblendd	$204, %ymm0, %ymm1, %ymm0
100005f03: c4 e3 fd 00 f0 d8           	vpermq	$216, %ymm0, %ymm6
100005f09: c4 e2 7d 21 fe              	vpmovsxbd	%xmm6, %ymm7
100005f0e: c4 e3 fd 00 c0 db           	vpermq	$219, %ymm0, %ymm0
100005f14: c4 a1 7a 6f 8c 71 0f fc ff ff       	vmovdqu	-1009(%rcx,%r14,2), %xmm1
100005f1e: c5 f9 6f 1d ca 12 00 00     	vmovdqa	4810(%rip), %xmm3
100005f26: c4 e2 71 00 cb              	vpshufb	%xmm3, %xmm1, %xmm1
100005f2b: c5 79 6f db                 	vmovdqa	%xmm3, %xmm11
100005f2f: 48 8b 44 24 38              	movq	56(%rsp), %rax
100005f34: c4 e2 79 78 28              	vpbroadcastb	(%rax), %xmm5
100005f39: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
100005f3e: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
100005f43: c4 62 7d 21 c5              	vpmovsxbd	%xmm5, %ymm8
100005f48: c4 e2 3d 40 c9              	vpmulld	%ymm1, %ymm8, %ymm1
100005f4d: c5 fd 7f 4c 24 40           	vmovdqa	%ymm1, 64(%rsp)
100005f53: c4 a1 7e 6f ac 71 20 fc ff ff       	vmovdqu	-992(%rcx,%r14,2), %ymm5
100005f5d: c4 e3 7d 39 f6 01           	vextracti128	$1, %ymm6, %xmm6
100005f63: c4 42 55 00 cf              	vpshufb	%ymm15, %ymm5, %ymm9
100005f68: c4 42 6d 00 d6              	vpshufb	%ymm14, %ymm2, %ymm10
100005f6d: c4 43 2d 02 c9 cc           	vpblendd	$204, %ymm9, %ymm10, %ymm9
100005f73: c4 43 fd 00 e9 d8           	vpermq	$216, %ymm9, %ymm13
100005f79: c5 fd 6f 0d df 13 00 00     	vmovdqa	5087(%rip), %ymm1
100005f81: c4 e2 55 00 e9              	vpshufb	%ymm1, %ymm5, %ymm5
100005f86: c4 62 7d 21 d6              	vpmovsxbd	%xmm6, %ymm10
100005f8b: c5 fd 6f 0d ed 13 00 00     	vmovdqa	5101(%rip), %ymm1
100005f93: c4 e2 6d 00 d1              	vpshufb	%ymm1, %ymm2, %ymm2
100005f98: c4 e3 6d 02 d5 cc           	vpblendd	$204, %ymm5, %ymm2, %ymm2
100005f9e: c4 e3 fd 00 ea d8           	vpermq	$216, %ymm2, %ymm5
100005fa4: c4 c2 7d 21 f5              	vpmovsxbd	%xmm13, %ymm6
100005fa9: c4 e2 3d 40 c0              	vpmulld	%ymm0, %ymm8, %ymm0
100005fae: c4 43 fd 00 c9 db           	vpermq	$219, %ymm9, %ymm9
100005fb4: c4 42 7d 21 c9              	vpmovsxbd	%xmm9, %ymm9
100005fb9: c4 63 7d 39 ec 01           	vextracti128	$1, %ymm13, %xmm4
100005fbf: c4 a1 7a 6f 9c 71 10 fc ff ff       	vmovdqu	-1008(%rcx,%r14,2), %xmm3
100005fc9: c4 e2 3d 40 cf              	vpmulld	%ymm7, %ymm8, %ymm1
100005fce: c5 fd 7f 8c 24 00 01 00 00  	vmovdqa	%ymm1, 256(%rsp)
100005fd7: c4 c2 61 00 fb              	vpshufb	%xmm11, %xmm3, %xmm7
100005fdc: c4 e2 7d 21 ff              	vpmovsxbd	%xmm7, %ymm7
100005fe1: c4 e2 79 78 48 01           	vpbroadcastb	1(%rax), %xmm1
100005fe7: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
100005fec: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
100005ff1: c4 42 75 40 e9              	vpmulld	%ymm9, %ymm1, %ymm13
100005ff6: c4 e2 75 40 ff              	vpmulld	%ymm7, %ymm1, %ymm7
100005ffb: c4 62 75 40 ce              	vpmulld	%ymm6, %ymm1, %ymm9
100006000: c4 e3 7d 39 ee 01           	vextracti128	$1, %ymm5, %xmm6
100006006: c4 e3 fd 00 d2 db           	vpermq	$219, %ymm2, %ymm2
10000600c: c4 62 7d 21 e2              	vpmovsxbd	%xmm2, %ymm12
100006011: c4 e2 7d 21 d5              	vpmovsxbd	%xmm5, %ymm2
100006016: c5 f9 6f 2d e2 11 00 00     	vmovdqa	4578(%rip), %xmm5
10000601e: c4 e2 61 00 dd              	vpshufb	%xmm5, %xmm3, %xmm3
100006023: c4 e2 7d 21 ee              	vpmovsxbd	%xmm6, %ymm5
100006028: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
10000602d: c4 e2 79 78 70 02           	vpbroadcastb	2(%rax), %xmm6
100006033: c4 62 7d 21 de              	vpmovsxbd	%xmm6, %ymm11
100006038: c4 e2 25 40 d2              	vpmulld	%ymm2, %ymm11, %ymm2
10000603d: c5 fd 7f 94 24 e0 02 00 00  	vmovdqa	%ymm2, 736(%rsp)
100006046: c4 42 25 40 e4              	vpmulld	%ymm12, %ymm11, %ymm12
10000604b: c4 e2 25 40 db              	vpmulld	%ymm3, %ymm11, %ymm3
100006050: c4 c2 3d 40 d2              	vpmulld	%ymm10, %ymm8, %ymm2
100006055: c5 fd 7f 94 24 e0 00 00 00  	vmovdqa	%ymm2, 224(%rsp)
10000605e: c4 21 7e 6f 84 71 ff fd ff ff       	vmovdqu	-513(%rcx,%r14,2), %ymm8
100006068: c4 21 7e 6f 94 71 1f fe ff ff       	vmovdqu	-481(%rcx,%r14,2), %ymm10
100006072: c4 42 2d 00 d7              	vpshufb	%ymm15, %ymm10, %ymm10
100006077: c4 42 3d 00 c6              	vpshufb	%ymm14, %ymm8, %ymm8
10000607c: c4 e2 75 40 cc              	vpmulld	%ymm4, %ymm1, %ymm1
100006081: c5 fd 7f 8c 24 40 01 00 00  	vmovdqa	%ymm1, 320(%rsp)
10000608a: c4 c3 3d 02 ca cc           	vpblendd	$204, %ymm10, %ymm8, %ymm1
100006090: c4 e3 fd 00 e1 d8           	vpermq	$216, %ymm1, %ymm4
100006096: c4 62 7d 21 c4              	vpmovsxbd	%xmm4, %ymm8
10000609b: c4 e3 fd 00 c9 db           	vpermq	$219, %ymm1, %ymm1
1000060a1: c4 e2 25 40 d5              	vpmulld	%ymm5, %ymm11, %ymm2
1000060a6: c5 fd 7f 94 24 60 01 00 00  	vmovdqa	%ymm2, 352(%rsp)
1000060af: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000060b4: c4 e3 7d 39 e4 01           	vextracti128	$1, %ymm4, %xmm4
1000060ba: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
1000060bf: c4 a1 7a 6f ac 71 0f fe ff ff       	vmovdqu	-497(%rcx,%r14,2), %xmm5
1000060c9: c4 e2 79 78 70 03           	vpbroadcastb	3(%rax), %xmm6
1000060cf: c5 f9 6f 15 19 11 00 00     	vmovdqa	4377(%rip), %xmm2
1000060d7: c4 e2 51 00 ea              	vpshufb	%xmm2, %xmm5, %xmm5
1000060dc: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
1000060e1: c4 62 4d 40 d9              	vpmulld	%ymm1, %ymm6, %ymm11
1000060e6: c4 e2 4d 40 cc              	vpmulld	%ymm4, %ymm6, %ymm1
1000060eb: c5 fd 7f 8c 24 20 01 00 00  	vmovdqa	%ymm1, 288(%rsp)
1000060f4: c4 e2 7d 21 cd              	vpmovsxbd	%xmm5, %ymm1
1000060f9: c4 c2 4d 40 e0              	vpmulld	%ymm8, %ymm6, %ymm4
1000060fe: c4 e2 4d 40 c9              	vpmulld	%ymm1, %ymm6, %ymm1
100006103: c4 41 7d fe ed              	vpaddd	%ymm13, %ymm0, %ymm13
100006108: c5 c5 fe 44 24 40           	vpaddd	64(%rsp), %ymm7, %ymm0
10000610e: c5 e5 fe c9                 	vpaddd	%ymm1, %ymm3, %ymm1
100006112: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
100006116: c5 fd 7f 44 24 40           	vmovdqa	%ymm0, 64(%rsp)
10000611c: c4 a1 7e 6f 84 71 00 fe ff ff       	vmovdqu	-512(%rcx,%r14,2), %ymm0
100006126: c4 a1 7e 6f 9c 71 20 fe ff ff       	vmovdqu	-480(%rcx,%r14,2), %ymm3
100006130: c5 35 fe 94 24 00 01 00 00  	vpaddd	256(%rsp), %ymm9, %ymm10
100006139: c4 c2 65 00 ef              	vpshufb	%ymm15, %ymm3, %ymm5
10000613e: c4 c2 7d 00 f6              	vpshufb	%ymm14, %ymm0, %ymm6
100006143: c4 e3 4d 02 ed cc           	vpblendd	$204, %ymm5, %ymm6, %ymm5
100006149: c4 e3 fd 00 f5 d8           	vpermq	$216, %ymm5, %ymm6
10000614f: c4 e2 65 00 1d 08 12 00 00  	vpshufb	4616(%rip), %ymm3, %ymm3
100006158: c4 c1 1d fe fb              	vpaddd	%ymm11, %ymm12, %ymm7
10000615d: c4 e2 7d 00 05 1a 12 00 00  	vpshufb	4634(%rip), %ymm0, %ymm0
100006166: c4 e3 7d 02 db cc           	vpblendd	$204, %ymm3, %ymm0, %ymm3
10000616c: c4 e3 7d 39 f0 01           	vextracti128	$1, %ymm6, %xmm0
100006172: c4 e3 fd 00 ed db           	vpermq	$219, %ymm5, %ymm5
100006178: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
10000617d: c5 5d fe 8c 24 e0 02 00 00  	vpaddd	736(%rsp), %ymm4, %ymm9
100006186: c4 e2 7d 21 e6              	vpmovsxbd	%xmm6, %ymm4
10000618b: c4 a1 7a 6f b4 71 10 fe ff ff       	vmovdqu	-496(%rcx,%r14,2), %xmm6
100006195: c4 e2 49 00 ca              	vpshufb	%xmm2, %xmm6, %xmm1
10000619a: c4 e2 79 78 50 04           	vpbroadcastb	4(%rax), %xmm2
1000061a0: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
1000061a5: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000061aa: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000061af: c4 e2 6d 40 e4              	vpmulld	%ymm4, %ymm2, %ymm4
1000061b4: c4 e2 6d 40 ed              	vpmulld	%ymm5, %ymm2, %ymm5
1000061b9: c4 62 6d 40 e0              	vpmulld	%ymm0, %ymm2, %ymm12
1000061be: c4 e2 6d 40 c9              	vpmulld	%ymm1, %ymm2, %ymm1
1000061c3: c4 e3 fd 00 d3 d8           	vpermq	$216, %ymm3, %ymm2
1000061c9: c4 e3 fd 00 db db           	vpermq	$219, %ymm3, %ymm3
1000061cf: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000061d4: c4 62 7d 21 da              	vpmovsxbd	%xmm2, %ymm11
1000061d9: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000061df: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000061e4: c4 e2 79 78 40 05           	vpbroadcastb	5(%rax), %xmm0
1000061ea: c4 e2 49 00 35 0d 10 00 00  	vpshufb	4109(%rip), %xmm6, %xmm6
1000061f3: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
1000061f8: c4 e2 7d 40 db              	vpmulld	%ymm3, %ymm0, %ymm3
1000061fd: c4 62 7d 40 c2              	vpmulld	%ymm2, %ymm0, %ymm8
100006202: c4 e2 7d 21 d6              	vpmovsxbd	%xmm6, %ymm2
100006207: c4 c2 7d 40 f3              	vpmulld	%ymm11, %ymm0, %ymm6
10000620c: c4 e2 7d 40 c2              	vpmulld	%ymm2, %ymm0, %ymm0
100006211: c5 d5 fe d3                 	vpaddd	%ymm3, %ymm5, %ymm2
100006215: c5 dd fe de                 	vpaddd	%ymm6, %ymm4, %ymm3
100006219: c5 f5 fe c0                 	vpaddd	%ymm0, %ymm1, %ymm0
10000621d: c4 a1 7e 6f 4c 71 ff        	vmovdqu	-1(%rcx,%r14,2), %ymm1
100006224: c4 a1 7e 6f 64 71 1f        	vmovdqu	31(%rcx,%r14,2), %ymm4
10000622b: c4 c2 5d 00 e7              	vpshufb	%ymm15, %ymm4, %ymm4
100006230: c5 95 fe ef                 	vpaddd	%ymm7, %ymm13, %ymm5
100006234: c4 c2 75 00 ce              	vpshufb	%ymm14, %ymm1, %ymm1
100006239: c4 e3 75 02 cc cc           	vpblendd	$204, %ymm4, %ymm1, %ymm1
10000623f: c4 e3 fd 00 e1 d8           	vpermq	$216, %ymm1, %ymm4
100006245: c4 e2 79 78 70 06           	vpbroadcastb	6(%rax), %xmm6
10000624b: c4 c1 2d fe f9              	vpaddd	%ymm9, %ymm10, %ymm7
100006250: c4 62 7d 21 cc              	vpmovsxbd	%xmm4, %ymm9
100006255: c4 e2 7d 21 f6              	vpmovsxbd	%xmm6, %ymm6
10000625a: c4 42 4d 40 c9              	vpmulld	%ymm9, %ymm6, %ymm9
10000625f: c4 c1 65 fe d9              	vpaddd	%ymm9, %ymm3, %ymm3
100006264: c5 c5 fe db                 	vpaddd	%ymm3, %ymm7, %ymm3
100006268: c5 fd 7f 9c 24 00 01 00 00  	vmovdqa	%ymm3, 256(%rsp)
100006271: c4 e3 7d 39 e3 01           	vextracti128	$1, %ymm4, %xmm3
100006277: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
10000627c: c4 e3 fd 00 c9 db           	vpermq	$219, %ymm1, %ymm1
100006282: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
100006287: c4 e2 4d 40 c9              	vpmulld	%ymm1, %ymm6, %ymm1
10000628c: c5 ed fe d1                 	vpaddd	%ymm1, %ymm2, %ymm2
100006290: c4 a1 7a 6f 4c 71 0f        	vmovdqu	15(%rcx,%r14,2), %xmm1
100006297: c5 79 6f 1d 51 0f 00 00     	vmovdqa	3921(%rip), %xmm11
10000629f: c4 c2 71 00 cb              	vpshufb	%xmm11, %xmm1, %xmm1
1000062a4: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000062a9: c4 62 4d 40 d3              	vpmulld	%ymm3, %ymm6, %ymm10
1000062ae: c4 e2 4d 40 c9              	vpmulld	%ymm1, %ymm6, %ymm1
1000062b3: c5 fd fe c1                 	vpaddd	%ymm1, %ymm0, %ymm0
1000062b7: c5 fd fe 44 24 40           	vpaddd	64(%rsp), %ymm0, %ymm0
1000062bd: c5 fd 7f 44 24 40           	vmovdqa	%ymm0, 64(%rsp)
1000062c3: c4 a1 7e 6f 04 71           	vmovdqu	(%rcx,%r14,2), %ymm0
1000062c9: c5 55 fe ea                 	vpaddd	%ymm2, %ymm5, %ymm13
1000062cd: c4 a1 7e 6f 54 71 20        	vmovdqu	32(%rcx,%r14,2), %ymm2
1000062d4: c4 c2 6d 00 e7              	vpshufb	%ymm15, %ymm2, %ymm4
1000062d9: c4 c2 7d 00 ee              	vpshufb	%ymm14, %ymm0, %ymm5
1000062de: c4 e3 55 02 e4 cc           	vpblendd	$204, %ymm4, %ymm5, %ymm4
1000062e4: c4 e3 fd 00 ec d8           	vpermq	$216, %ymm4, %ymm5
1000062ea: c5 fd 6f 8c 24 40 01 00 00  	vmovdqa	320(%rsp), %ymm1
1000062f3: c5 f5 fe b4 24 e0 00 00 00  	vpaddd	224(%rsp), %ymm1, %ymm6
1000062fc: c4 e2 6d 00 15 5b 10 00 00  	vpshufb	4187(%rip), %ymm2, %ymm2
100006305: c4 e2 7d 00 05 72 10 00 00  	vpshufb	4210(%rip), %ymm0, %ymm0
10000630e: c4 e3 7d 02 c2 cc           	vpblendd	$204, %ymm2, %ymm0, %ymm0
100006314: c4 e3 fd 00 d0 d8           	vpermq	$216, %ymm0, %ymm2
10000631a: c4 e2 7d 21 fd              	vpmovsxbd	%xmm5, %ymm7
10000631f: c5 fd 6f 8c 24 20 01 00 00  	vmovdqa	288(%rsp), %ymm1
100006328: c5 75 fe 8c 24 60 01 00 00  	vpaddd	352(%rsp), %ymm1, %ymm9
100006331: c4 e3 fd 00 e4 db           	vpermq	$219, %ymm4, %ymm4
100006337: c4 e2 7d 21 e4              	vpmovsxbd	%xmm4, %ymm4
10000633c: c4 e3 7d 39 ed 01           	vextracti128	$1, %ymm5, %xmm5
100006342: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006347: c4 a1 7a 6f 4c 71 10        	vmovdqu	16(%rcx,%r14,2), %xmm1
10000634e: c4 41 1d fe c0              	vpaddd	%ymm8, %ymm12, %ymm8
100006353: c4 e2 79 78 58 07           	vpbroadcastb	7(%rax), %xmm3
100006359: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
10000635e: c4 62 65 40 e5              	vpmulld	%ymm5, %ymm3, %ymm12
100006363: c4 c1 4d fe f1              	vpaddd	%ymm9, %ymm6, %ymm6
100006368: c4 c2 71 00 eb              	vpshufb	%xmm11, %xmm1, %xmm5
10000636d: c4 e2 65 40 e4              	vpmulld	%ymm4, %ymm3, %ymm4
100006372: c4 e2 65 40 ff              	vpmulld	%ymm7, %ymm3, %ymm7
100006377: c4 41 3d fe c2              	vpaddd	%ymm10, %ymm8, %ymm8
10000637c: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006381: c4 e2 65 40 dd              	vpmulld	%ymm5, %ymm3, %ymm3
100006386: c4 e2 79 78 68 08           	vpbroadcastb	8(%rax), %xmm5
10000638c: c4 62 7d 21 ca              	vpmovsxbd	%xmm2, %ymm9
100006391: c4 e2 7d 21 ed              	vpmovsxbd	%xmm5, %ymm5
100006396: c4 42 55 40 c9              	vpmulld	%ymm9, %ymm5, %ymm9
10000639b: c4 c1 45 fe f9              	vpaddd	%ymm9, %ymm7, %ymm7
1000063a0: c4 e3 fd 00 c0 db           	vpermq	$219, %ymm0, %ymm0
1000063a6: c4 e2 7d 21 c0              	vpmovsxbd	%xmm0, %ymm0
1000063ab: c4 e2 55 40 c0              	vpmulld	%ymm0, %ymm5, %ymm0
1000063b0: c5 dd fe c0                 	vpaddd	%ymm0, %ymm4, %ymm0
1000063b4: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000063ba: c4 e2 71 00 0d 3d 0e 00 00  	vpshufb	3645(%rip), %xmm1, %xmm1
1000063c3: c4 e2 7d 21 d2              	vpmovsxbd	%xmm2, %ymm2
1000063c8: c4 e2 7d 21 c9              	vpmovsxbd	%xmm1, %ymm1
1000063cd: c4 e2 55 40 d2              	vpmulld	%ymm2, %ymm5, %ymm2
1000063d2: c4 e2 55 40 c9              	vpmulld	%ymm1, %ymm5, %ymm1
1000063d7: c4 c1 4d fe e0              	vpaddd	%ymm8, %ymm6, %ymm4
1000063dc: c5 9d fe d2                 	vpaddd	%ymm2, %ymm12, %ymm2
1000063e0: c5 e5 fe c9                 	vpaddd	%ymm1, %ymm3, %ymm1
1000063e4: c4 c2 79 78 1a              	vpbroadcastb	(%r10), %xmm3
1000063e9: c4 e2 7d 21 db              	vpmovsxbd	%xmm3, %ymm3
1000063ee: c5 c5 fe eb                 	vpaddd	%ymm3, %ymm7, %ymm5
1000063f2: c5 d5 fe ac 24 00 01 00 00  	vpaddd	256(%rsp), %ymm5, %ymm5
1000063fb: c5 fd fe c3                 	vpaddd	%ymm3, %ymm0, %ymm0
1000063ff: c5 95 fe c0                 	vpaddd	%ymm0, %ymm13, %ymm0
100006403: c5 ed fe d3                 	vpaddd	%ymm3, %ymm2, %ymm2
100006407: c5 f5 fe cb                 	vpaddd	%ymm3, %ymm1, %ymm1
10000640b: c5 dd fe d2                 	vpaddd	%ymm2, %ymm4, %ymm2
10000640f: c5 fd 6f bc 24 80 02 00 00  	vmovdqa	640(%rsp), %ymm7
100006418: c4 e2 7d 40 c7              	vpmulld	%ymm7, %ymm0, %ymm0
10000641d: c4 e2 55 40 df              	vpmulld	%ymm7, %ymm5, %ymm3
100006422: c4 e2 6d 40 d7              	vpmulld	%ymm7, %ymm2, %ymm2
100006427: c5 f5 fe 4c 24 40           	vpaddd	64(%rsp), %ymm1, %ymm1
10000642d: c5 fd 70 e3 f5              	vpshufd	$245, %ymm3, %ymm4
100006432: c4 e2 7d 58 2d e5 28 00 00  	vpbroadcastd	10469(%rip), %ymm5
10000643b: c4 e2 5d 28 e5              	vpmuldq	%ymm5, %ymm4, %ymm4
100006440: c4 e2 65 28 f5              	vpmuldq	%ymm5, %ymm3, %ymm6
100006445: c5 fd 70 f6 f5              	vpshufd	$245, %ymm6, %ymm6
10000644a: c4 e3 4d 02 e4 aa           	vpblendd	$170, %ymm4, %ymm6, %ymm4
100006450: c5 dd fe db                 	vpaddd	%ymm3, %ymm4, %ymm3
100006454: c5 fd 70 e0 f5              	vpshufd	$245, %ymm0, %ymm4
100006459: c4 e2 5d 28 e5              	vpmuldq	%ymm5, %ymm4, %ymm4
10000645e: c4 e2 7d 28 f5              	vpmuldq	%ymm5, %ymm0, %ymm6
100006463: c5 fd 70 f6 f5              	vpshufd	$245, %ymm6, %ymm6
100006468: c4 e3 4d 02 e4 aa           	vpblendd	$170, %ymm4, %ymm6, %ymm4
10000646e: c5 cd 72 d3 1f              	vpsrld	$31, %ymm3, %ymm6
100006473: c5 e5 72 e3 0d              	vpsrad	$13, %ymm3, %ymm3
100006478: c5 dd fe c0                 	vpaddd	%ymm0, %ymm4, %ymm0
10000647c: c5 dd 72 d0 1f              	vpsrld	$31, %ymm0, %ymm4
100006481: c5 e5 fe de                 	vpaddd	%ymm6, %ymm3, %ymm3
100006485: c5 fd 72 e0 0d              	vpsrad	$13, %ymm0, %ymm0
10000648a: c5 fd fe c4                 	vpaddd	%ymm4, %ymm0, %ymm0
10000648e: c5 fd 70 e2 f5              	vpshufd	$245, %ymm2, %ymm4
100006493: c4 e2 5d 28 e5              	vpmuldq	%ymm5, %ymm4, %ymm4
100006498: c4 e2 6d 28 f5              	vpmuldq	%ymm5, %ymm2, %ymm6
10000649d: c5 fd 70 f6 f5              	vpshufd	$245, %ymm6, %ymm6
1000064a2: c4 e3 4d 02 e4 aa           	vpblendd	$170, %ymm4, %ymm6, %ymm4
1000064a8: c4 e2 75 40 cf              	vpmulld	%ymm7, %ymm1, %ymm1
1000064ad: c5 dd fe d2                 	vpaddd	%ymm2, %ymm4, %ymm2
1000064b1: c5 fd 70 e1 f5              	vpshufd	$245, %ymm1, %ymm4
1000064b6: c4 e2 5d 28 e5              	vpmuldq	%ymm5, %ymm4, %ymm4
1000064bb: c4 e2 75 28 ed              	vpmuldq	%ymm5, %ymm1, %ymm5
1000064c0: c5 fd 70 ed f5              	vpshufd	$245, %ymm5, %ymm5
1000064c5: c4 e3 55 02 e4 aa           	vpblendd	$170, %ymm4, %ymm5, %ymm4
1000064cb: c5 d5 72 d2 1f              	vpsrld	$31, %ymm2, %ymm5
1000064d0: c5 ed 72 e2 0d              	vpsrad	$13, %ymm2, %ymm2
1000064d5: c5 ed fe d5                 	vpaddd	%ymm5, %ymm2, %ymm2
1000064d9: c5 dd fe c9                 	vpaddd	%ymm1, %ymm4, %ymm1
1000064dd: c5 dd 72 d1 1f              	vpsrld	$31, %ymm1, %ymm4
1000064e2: c5 f5 72 e1 0d              	vpsrad	$13, %ymm1, %ymm1
1000064e7: c5 f5 fe cc                 	vpaddd	%ymm4, %ymm1, %ymm1
1000064eb: c4 e2 7d 58 25 30 28 00 00  	vpbroadcastd	10288(%rip), %ymm4
1000064f4: c4 e2 6d 39 d4              	vpminsd	%ymm4, %ymm2, %ymm2
1000064f9: c4 e2 7d 39 c4              	vpminsd	%ymm4, %ymm0, %ymm0
1000064fe: c4 e2 65 39 dc              	vpminsd	%ymm4, %ymm3, %ymm3
100006503: c4 e2 75 39 cc              	vpminsd	%ymm4, %ymm1, %ymm1
100006508: c4 e2 7d 58 25 17 28 00 00  	vpbroadcastd	10263(%rip), %ymm4
100006511: c4 e2 7d 3d c4              	vpmaxsd	%ymm4, %ymm0, %ymm0
100006516: c4 e2 6d 3d d4              	vpmaxsd	%ymm4, %ymm2, %ymm2
10000651b: c5 ed 6b c0                 	vpackssdw	%ymm0, %ymm2, %ymm0
10000651f: c4 e2 65 3d d4              	vpmaxsd	%ymm4, %ymm3, %ymm2
100006524: c4 e2 75 3d cc              	vpmaxsd	%ymm4, %ymm1, %ymm1
100006529: c5 ed 6b c9                 	vpackssdw	%ymm1, %ymm2, %ymm1
10000652d: c5 fd 6f b4 24 60 02 00 00  	vmovdqa	608(%rsp), %ymm6
100006536: c5 ed 73 d6 01              	vpsrlq	$1, %ymm6, %ymm2
10000653b: c5 fd 6f ac 24 c0 02 00 00  	vmovdqa	704(%rsp), %ymm5
100006544: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006548: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000654d: c5 fd 6f a4 24 a0 02 00 00  	vmovdqa	672(%rsp), %ymm4
100006556: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
10000655a: c4 c1 f9 7e d7              	vmovq	%xmm2, %r15
10000655f: c4 e3 f9 16 d2 01           	vpextrq	$1, %xmm2, %rdx
100006565: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
10000656b: c4 c1 f9 7e d0              	vmovq	%xmm2, %r8
100006570: c4 c3 f9 16 d1 01           	vpextrq	$1, %xmm2, %r9
100006576: c5 fd 6f bc 24 40 02 00 00  	vmovdqa	576(%rsp), %ymm7
10000657f: c5 ed 73 d7 01              	vpsrlq	$1, %ymm7, %ymm2
100006584: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006588: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000658d: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006591: c4 e1 f9 7e d7              	vmovq	%xmm2, %rdi
100006596: c4 c3 f9 16 d2 01           	vpextrq	$1, %xmm2, %r10
10000659c: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000065a2: c5 f9 d6 94 24 e0 00 00 00  	vmovq	%xmm2, 224(%rsp)
1000065ab: c4 e3 f9 16 d3 01           	vpextrq	$1, %xmm2, %rbx
1000065b1: c5 7d 6f 84 24 20 02 00 00  	vmovdqa	544(%rsp), %ymm8
1000065ba: c4 c1 6d 73 d0 01           	vpsrlq	$1, %ymm8, %ymm2
1000065c0: c4 e3 fd 00 c0 d8           	vpermq	$216, %ymm0, %ymm0
1000065c6: c4 e3 fd 00 c9 d8           	vpermq	$216, %ymm1, %ymm1
1000065cc: c5 f5 63 c0                 	vpacksswb	%ymm0, %ymm1, %ymm0
1000065d0: c5 7d 6f 8c 24 00 02 00 00  	vmovdqa	512(%rsp), %ymm9
1000065d9: c4 c1 65 73 d1 01           	vpsrlq	$1, %ymm9, %ymm3
1000065df: c5 ed d4 cd                 	vpaddq	%ymm5, %ymm2, %ymm1
1000065e3: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
1000065e8: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
1000065ec: c4 e3 f9 16 8c 24 60 01 00 00 01    	vpextrq	$1, %xmm1, 352(%rsp)
1000065f7: c4 e1 f9 7e c8              	vmovq	%xmm1, %rax
1000065fc: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
100006602: c4 e1 f9 7e c9              	vmovq	%xmm1, %rcx
100006607: c4 c3 f9 16 cd 01           	vpextrq	$1, %xmm1, %r13
10000660d: c5 7d 6f 94 24 e0 01 00 00  	vmovdqa	480(%rsp), %ymm10
100006616: c4 c1 75 73 d2 01           	vpsrlq	$1, %ymm10, %ymm1
10000661c: c5 e5 d4 d5                 	vpaddq	%ymm5, %ymm3, %ymm2
100006620: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006625: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006629: c4 83 79 14 04 3c 00        	vpextrb	$0, %xmm0, (%r12,%r15)
100006630: c4 e1 f9 7e d6              	vmovq	%xmm2, %rsi
100006635: c4 c3 f9 16 d7 01           	vpextrq	$1, %xmm2, %r15
10000663b: c4 c3 79 14 04 14 01        	vpextrb	$1, %xmm0, (%r12,%rdx)
100006642: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006648: c4 e3 f9 16 94 24 40 01 00 00 01    	vpextrq	$1, %xmm2, 320(%rsp)
100006653: c4 83 79 14 04 04 02        	vpextrb	$2, %xmm0, (%r12,%r8)
10000665a: c4 c1 f9 7e d0              	vmovq	%xmm2, %r8
10000665f: c5 fd 6f 9c 24 c0 01 00 00  	vmovdqa	448(%rsp), %ymm3
100006668: c5 ed 73 d3 01              	vpsrlq	$1, %ymm3, %ymm2
10000666d: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006671: c5 f5 d4 cd                 	vpaddq	%ymm5, %ymm1, %ymm1
100006675: c5 f5 73 f1 03              	vpsllq	$3, %ymm1, %ymm1
10000667a: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
10000667f: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006683: c5 f5 d4 cc                 	vpaddq	%ymm4, %ymm1, %ymm1
100006687: c4 83 79 14 04 0c 03        	vpextrb	$3, %xmm0, (%r12,%r9)
10000668e: c4 c1 f9 7e cb              	vmovq	%xmm1, %r11
100006693: c4 c3 79 14 04 3c 04        	vpextrb	$4, %xmm0, (%r12,%rdi)
10000669a: c4 e3 f9 16 8c 24 20 01 00 00 01    	vpextrq	$1, %xmm1, 288(%rsp)
1000066a5: c4 e3 7d 39 c9 01           	vextracti128	$1, %ymm1, %xmm1
1000066ab: c4 83 79 14 04 14 05        	vpextrb	$5, %xmm0, (%r12,%r10)
1000066b2: 48 8b 94 24 e0 00 00 00     	movq	224(%rsp), %rdx
1000066ba: c4 c3 79 14 04 14 06        	vpextrb	$6, %xmm0, (%r12,%rdx)
1000066c1: c4 c1 f9 7e ca              	vmovq	%xmm1, %r10
1000066c6: c4 e3 f9 16 4c 24 40 01     	vpextrq	$1, %xmm1, 64(%rsp)
1000066ce: c4 c3 79 14 04 1c 07        	vpextrb	$7, %xmm0, (%r12,%rbx)
1000066d5: c4 e3 f9 16 94 24 e0 00 00 00 01    	vpextrq	$1, %xmm2, 224(%rsp)
1000066e0: c4 e3 7d 39 c1 01           	vextracti128	$1, %ymm0, %xmm1
1000066e6: c4 c3 79 14 0c 04 00        	vpextrb	$0, %xmm1, (%r12,%rax)
1000066ed: c4 e1 f9 7e d7              	vmovq	%xmm2, %rdi
1000066f2: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000066f8: 48 8b 84 24 60 01 00 00     	movq	352(%rsp), %rax
100006700: c4 c3 79 14 0c 04 01        	vpextrb	$1, %xmm1, (%r12,%rax)
100006707: c4 c1 f9 7e d1              	vmovq	%xmm2, %r9
10000670c: c4 c3 79 14 0c 0c 02        	vpextrb	$2, %xmm1, (%r12,%rcx)
100006713: c4 e3 f9 16 d2 01           	vpextrq	$1, %xmm2, %rdx
100006719: c5 7d 6f 9c 24 a0 01 00 00  	vmovdqa	416(%rsp), %ymm11
100006722: c4 c1 6d 73 d3 01           	vpsrlq	$1, %ymm11, %ymm2
100006728: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
10000672c: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006731: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006735: c4 83 79 14 0c 2c 03        	vpextrb	$3, %xmm1, (%r12,%r13)
10000673c: c4 e1 f9 7e d1              	vmovq	%xmm2, %rcx
100006741: c4 c3 79 14 0c 34 04        	vpextrb	$4, %xmm1, (%r12,%rsi)
100006748: c4 e3 f9 16 d6 01           	vpextrq	$1, %xmm2, %rsi
10000674e: c4 83 79 14 0c 3c 05        	vpextrb	$5, %xmm1, (%r12,%r15)
100006755: c4 83 79 14 0c 04 06        	vpextrb	$6, %xmm1, (%r12,%r8)
10000675c: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
100006762: c4 c1 f9 7e d0              	vmovq	%xmm2, %r8
100006767: c4 c3 f9 16 d7 01           	vpextrq	$1, %xmm2, %r15
10000676d: c5 7d 6f a4 24 80 01 00 00  	vmovdqa	384(%rsp), %ymm12
100006776: c4 c1 6d 73 d4 01           	vpsrlq	$1, %ymm12, %ymm2
10000677c: c5 ed d4 d5                 	vpaddq	%ymm5, %ymm2, %ymm2
100006780: c5 ed 73 f2 03              	vpsllq	$3, %ymm2, %ymm2
100006785: c5 ed d4 d4                 	vpaddq	%ymm4, %ymm2, %ymm2
100006789: 48 8b 84 24 40 01 00 00     	movq	320(%rsp), %rax
100006791: c4 c3 79 14 0c 04 07        	vpextrb	$7, %xmm1, (%r12,%rax)
100006798: c4 e1 f9 7e d0              	vmovq	%xmm2, %rax
10000679d: c4 83 79 14 04 1c 08        	vpextrb	$8, %xmm0, (%r12,%r11)
1000067a4: c4 c3 f9 16 d3 01           	vpextrq	$1, %xmm2, %r11
1000067aa: c4 e3 7d 39 d2 01           	vextracti128	$1, %ymm2, %xmm2
1000067b0: 48 8b 9c 24 20 01 00 00     	movq	288(%rsp), %rbx
1000067b8: c4 c3 79 14 04 1c 09        	vpextrb	$9, %xmm0, (%r12,%rbx)
1000067bf: c4 83 79 14 04 14 0a        	vpextrb	$10, %xmm0, (%r12,%r10)
1000067c6: c4 c1 f9 7e d2              	vmovq	%xmm2, %r10
1000067cb: c4 c3 f9 16 d5 01           	vpextrq	$1, %xmm2, %r13
1000067d1: 48 8b 5c 24 40              	movq	64(%rsp), %rbx
1000067d6: c4 c3 79 14 04 1c 0b        	vpextrb	$11, %xmm0, (%r12,%rbx)
1000067dd: c4 c3 79 14 04 3c 0c        	vpextrb	$12, %xmm0, (%r12,%rdi)
1000067e4: 48 8b bc 24 e0 00 00 00     	movq	224(%rsp), %rdi
1000067ec: c4 c3 79 14 04 3c 0d        	vpextrb	$13, %xmm0, (%r12,%rdi)
1000067f3: c4 83 79 14 04 0c 0e        	vpextrb	$14, %xmm0, (%r12,%r9)
1000067fa: c4 c3 79 14 04 14 0f        	vpextrb	$15, %xmm0, (%r12,%rdx)
100006801: c4 c3 79 14 0c 0c 08        	vpextrb	$8, %xmm1, (%r12,%rcx)
100006808: c4 c3 79 14 0c 34 09        	vpextrb	$9, %xmm1, (%r12,%rsi)
10000680f: c4 83 79 14 0c 04 0a        	vpextrb	$10, %xmm1, (%r12,%r8)
100006816: c4 83 79 14 0c 3c 0b        	vpextrb	$11, %xmm1, (%r12,%r15)
10000681d: c4 c3 79 14 0c 04 0c        	vpextrb	$12, %xmm1, (%r12,%rax)
100006824: c4 83 79 14 0c 1c 0d        	vpextrb	$13, %xmm1, (%r12,%r11)
10000682b: c4 83 79 14 0c 14 0e        	vpextrb	$14, %xmm1, (%r12,%r10)
100006832: 4c 8b 54 24 10              	movq	16(%rsp), %r10
100006837: c4 83 79 14 0c 2c 0f        	vpextrb	$15, %xmm1, (%r12,%r13)
10000683e: c4 e2 7d 59 05 e9 24 00 00  	vpbroadcastq	9449(%rip), %ymm0
100006847: c5 cd d4 f0                 	vpaddq	%ymm0, %ymm6, %ymm6
10000684b: c5 fd 7f b4 24 60 02 00 00  	vmovdqa	%ymm6, 608(%rsp)
100006854: c5 c5 d4 f8                 	vpaddq	%ymm0, %ymm7, %ymm7
100006858: c5 fd 7f bc 24 40 02 00 00  	vmovdqa	%ymm7, 576(%rsp)
100006861: c5 3d d4 c0                 	vpaddq	%ymm0, %ymm8, %ymm8
100006865: c5 7d 7f 84 24 20 02 00 00  	vmovdqa	%ymm8, 544(%rsp)
10000686e: c5 35 d4 c8                 	vpaddq	%ymm0, %ymm9, %ymm9
100006872: c5 7d 7f 8c 24 00 02 00 00  	vmovdqa	%ymm9, 512(%rsp)
10000687b: c5 2d d4 d0                 	vpaddq	%ymm0, %ymm10, %ymm10
10000687f: c5 7d 7f 94 24 e0 01 00 00  	vmovdqa	%ymm10, 480(%rsp)
100006888: c5 e5 d4 d8                 	vpaddq	%ymm0, %ymm3, %ymm3
10000688c: c5 fd 7f 9c 24 c0 01 00 00  	vmovdqa	%ymm3, 448(%rsp)
100006895: c5 25 d4 d8                 	vpaddq	%ymm0, %ymm11, %ymm11
100006899: c5 7d 7f 9c 24 a0 01 00 00  	vmovdqa	%ymm11, 416(%rsp)
1000068a2: c5 1d d4 e0                 	vpaddq	%ymm0, %ymm12, %ymm12
1000068a6: c5 7d 7f a4 24 80 01 00 00  	vmovdqa	%ymm12, 384(%rsp)
1000068af: 49 83 c6 20                 	addq	$32, %r14
1000068b3: 49 81 fe e0 00 00 00        	cmpq	$224, %r14
1000068ba: 0f 85 10 f6 ff ff           	jne	-2544 <__ZN11LineNetwork7forwardEv+0x1e20>
1000068c0: b9 c0 01 00 00              	movl	$448, %ecx
1000068c5: 44 8b 44 24 24              	movl	36(%rsp), %r8d
1000068ca: 48 8b b4 24 98 00 00 00     	movq	152(%rsp), %rsi
1000068d2: 41 bf 7f 00 00 00           	movl	$127, %r15d
1000068d8: 41 bd 81 00 00 00           	movl	$129, %r13d
1000068de: eb 07                       	jmp	7 <__ZN11LineNetwork7forwardEv+0x2837>
1000068e0: 31 c9                       	xorl	%ecx, %ecx
1000068e2: 4c 8b 54 24 10              	movq	16(%rsp), %r10
1000068e7: 48 83 44 24 18 02           	addq	$2, 24(%rsp)
1000068ed: 48 89 c8                    	movq	%rcx, %rax
1000068f0: 48 d1 e8                    	shrq	%rax
1000068f3: 4c 8b 9c 24 d8 00 00 00     	movq	216(%rsp), %r11
1000068fb: 4c 01 d8                    	addq	%r11, %rax
1000068fe: 48 8b 54 24 28              	movq	40(%rsp), %rdx
100006903: 4c 8d 0c c2                 	leaq	(%rdx,%rax,8), %r9
100006907: 4c 8b b4 24 d0 00 00 00     	movq	208(%rsp), %r14
10000690f: 90                          	nop
100006910: 41 0f be 94 0e fe fb ff ff  	movsbl	-1026(%r14,%rcx), %edx
100006919: 48 8b 44 24 38              	movq	56(%rsp), %rax
10000691e: 0f be 18                    	movsbl	(%rax), %ebx
100006921: 0f af da                    	imull	%edx, %ebx
100006924: 41 0f be 94 0e ff fb ff ff  	movsbl	-1025(%r14,%rcx), %edx
10000692d: 0f be 78 01                 	movsbl	1(%rax), %edi
100006931: 0f af fa                    	imull	%edx, %edi
100006934: 01 df                       	addl	%ebx, %edi
100006936: 41 0f be 94 0e 00 fc ff ff  	movsbl	-1024(%r14,%rcx), %edx
10000693f: 0f be 58 02                 	movsbl	2(%rax), %ebx
100006943: 0f af da                    	imull	%edx, %ebx
100006946: 01 fb                       	addl	%edi, %ebx
100006948: 41 0f be 94 0e fe fd ff ff  	movsbl	-514(%r14,%rcx), %edx
100006951: 0f be 78 03                 	movsbl	3(%rax), %edi
100006955: 0f af fa                    	imull	%edx, %edi
100006958: 01 df                       	addl	%ebx, %edi
10000695a: 41 0f be 94 0e ff fd ff ff  	movsbl	-513(%r14,%rcx), %edx
100006963: 0f be 58 04                 	movsbl	4(%rax), %ebx
100006967: 0f af da                    	imull	%edx, %ebx
10000696a: 01 fb                       	addl	%edi, %ebx
10000696c: 41 0f be 94 0e 00 fe ff ff  	movsbl	-512(%r14,%rcx), %edx
100006975: 0f be 78 05                 	movsbl	5(%rax), %edi
100006979: 0f af fa                    	imull	%edx, %edi
10000697c: 01 df                       	addl	%ebx, %edi
10000697e: 41 0f be 54 0e fe           	movsbl	-2(%r14,%rcx), %edx
100006984: 0f be 58 06                 	movsbl	6(%rax), %ebx
100006988: 0f af da                    	imull	%edx, %ebx
10000698b: 01 fb                       	addl	%edi, %ebx
10000698d: 41 0f be 54 0e ff           	movsbl	-1(%r14,%rcx), %edx
100006993: 0f be 78 07                 	movsbl	7(%rax), %edi
100006997: 0f af fa                    	imull	%edx, %edi
10000699a: 01 df                       	addl	%ebx, %edi
10000699c: 41 0f be 14 0e              	movsbl	(%r14,%rcx), %edx
1000069a1: 0f be 58 08                 	movsbl	8(%rax), %ebx
1000069a5: 0f af da                    	imull	%edx, %ebx
1000069a8: 01 fb                       	addl	%edi, %ebx
1000069aa: 41 0f be 12                 	movsbl	(%r10), %edx
1000069ae: 01 da                       	addl	%ebx, %edx
1000069b0: 41 0f af d0                 	imull	%r8d, %edx
1000069b4: 48 63 d2                    	movslq	%edx, %rdx
1000069b7: 48 69 fa 09 04 02 81        	imulq	$-2130574327, %rdx, %rdi
1000069be: 48 c1 ef 20                 	shrq	$32, %rdi
1000069c2: 01 fa                       	addl	%edi, %edx
1000069c4: 89 d7                       	movl	%edx, %edi
1000069c6: c1 ef 1f                    	shrl	$31, %edi
1000069c9: c1 fa 0d                    	sarl	$13, %edx
1000069cc: 01 fa                       	addl	%edi, %edx
1000069ce: 81 fa 80 00 00 00           	cmpl	$128, %edx
1000069d4: 41 0f 4d d7                 	cmovgel	%r15d, %edx
1000069d8: 83 fa 81                    	cmpl	$-127, %edx
1000069db: 41 0f 4e d5                 	cmovlel	%r13d, %edx
1000069df: 41 88 11                    	movb	%dl, (%r9)
1000069e2: 48 83 c1 02                 	addq	$2, %rcx
1000069e6: 49 83 c1 08                 	addq	$8, %r9
1000069ea: 48 81 f9 fd 01 00 00        	cmpq	$509, %rcx
1000069f1: 0f 82 19 ff ff ff           	jb	-231 <__ZN11LineNetwork7forwardEv+0x2860>
1000069f7: 48 8b 8c 24 c8 00 00 00     	movq	200(%rsp), %rcx
1000069ff: 48 ff c1                    	incq	%rcx
100006a02: 48 8b 5c 24 30              	movq	48(%rsp), %rbx
100006a07: 48 81 c3 00 04 00 00        	addq	$1024, %rbx
100006a0e: 49 81 c6 00 04 00 00        	addq	$1024, %r14
100006a15: 49 81 c3 00 01 00 00        	addq	$256, %r11
100006a1c: 48 81 7c 24 18 fd 01 00 00  	cmpq	$509, 24(%rsp)
100006a25: 0f 82 45 f3 ff ff           	jb	-3259 <__ZN11LineNetwork7forwardEv+0x1cc0>
100006a2b: 48 89 f3                    	movq	%rsi, %rbx
100006a2e: 48 ff c3                    	incq	%rbx
100006a31: 48 ff 44 24 28              	incq	40(%rsp)
100006a36: 48 83 fb 08                 	cmpq	$8, %rbx
100006a3a: 0f 85 90 f2 ff ff           	jne	-3440 <__ZN11LineNetwork7forwardEv+0x1c20>
100006a40: 48 8d 65 d8                 	leaq	-40(%rbp), %rsp
100006a44: 5b                          	popq	%rbx
100006a45: 41 5c                       	popq	%r12
100006a47: 41 5d                       	popq	%r13
100006a49: 41 5e                       	popq	%r14
100006a4b: 41 5f                       	popq	%r15
100006a4d: 5d                          	popq	%rbp
100006a4e: c5 f8 77                    	vzeroupper
100006a51: c3                          	retq
100006a52: 66 2e 0f 1f 84 00 00 00 00 00       	nopw	%cs:(%rax,%rax)
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
