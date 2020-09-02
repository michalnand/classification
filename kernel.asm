0000a09c <_Z15dot_microkernelILj48EaasET2_PKT0_PKT1_>:
    a09c:	e92d 4ff0 	stmdb	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, lr}
    a0a0:	460a      	mov	r2, r1
    a0a2:	b0b1      	sub	sp, #196	; 0xc4
    a0a4:	4607      	mov	r7, r0
    a0a6:	f992 c004 	ldrsb.w	ip, [r2, #4]
    a0aa:	f8cd c01c 	str.w	ip, [sp, #28]
    a0ae:	f997 c005 	ldrsb.w	ip, [r7, #5]
    a0b2:	f8cd c020 	str.w	ip, [sp, #32]
    a0b6:	f992 c005 	ldrsb.w	ip, [r2, #5]
    a0ba:	f8cd c024 	str.w	ip, [sp, #36]	; 0x24
    a0be:	f997 c006 	ldrsb.w	ip, [r7, #6]
    a0c2:	f8cd c028 	str.w	ip, [sp, #40]	; 0x28
    a0c6:	f992 c006 	ldrsb.w	ip, [r2, #6]
    a0ca:	f992 4001 	ldrsb.w	r4, [r2, #1]
    a0ce:	f992 5002 	ldrsb.w	r5, [r2, #2]
    a0d2:	f8cd c02c 	str.w	ip, [sp, #44]	; 0x2c
    a0d6:	f997 c007 	ldrsb.w	ip, [r7, #7]
    a0da:	f997 3001 	ldrsb.w	r3, [r7, #1]
    a0de:	f997 6003 	ldrsb.w	r6, [r7, #3]
    a0e2:	9402      	str	r4, [sp, #8]
    a0e4:	2000      	movs	r0, #0
    a0e6:	2100      	movs	r1, #0
    a0e8:	f992 4003 	ldrsb.w	r4, [r2, #3]
    a0ec:	9503      	str	r5, [sp, #12]
    a0ee:	f8cd c034 	str.w	ip, [sp, #52]	; 0x34
    a0f2:	f997 5004 	ldrsb.w	r5, [r7, #4]
    a0f6:	f992 c007 	ldrsb.w	ip, [r2, #7]
    a0fa:	f997 8000 	ldrsb.w	r8, [r7]
    a0fe:	f997 b002 	ldrsb.w	fp, [r7, #2]
    a102:	f992 e000 	ldrsb.w	lr, [r2]
    a106:	9301      	str	r3, [sp, #4]
    a108:	e9cd 0120 	strd	r0, r1, [sp, #128]	; 0x80
    a10c:	e9cd 0122 	strd	r0, r1, [sp, #136]	; 0x88
    a110:	9604      	str	r6, [sp, #16]
    a112:	9405      	str	r4, [sp, #20]
    a114:	9506      	str	r5, [sp, #24]
    a116:	f8cd c038 	str.w	ip, [sp, #56]	; 0x38
    a11a:	f997 c008 	ldrsb.w	ip, [r7, #8]
    a11e:	f8cd c040 	str.w	ip, [sp, #64]	; 0x40
    a122:	e9cd 0128 	strd	r0, r1, [sp, #160]	; 0xa0
    a126:	f992 c008 	ldrsb.w	ip, [r2, #8]
    a12a:	f997 a00a 	ldrsb.w	sl, [r7, #10]
    a12e:	f8cd c044 	str.w	ip, [sp, #68]	; 0x44
    a132:	e9cd 0124 	strd	r0, r1, [sp, #144]	; 0x90
    a136:	f997 c009 	ldrsb.w	ip, [r7, #9]
    a13a:	f8cd a050 	str.w	sl, [sp, #80]	; 0x50
    a13e:	e9cd 0126 	strd	r0, r1, [sp, #152]	; 0x98
    a142:	f997 a012 	ldrsb.w	sl, [r7, #18]
    a146:	f8cd c048 	str.w	ip, [sp, #72]	; 0x48
    a14a:	e9cd 012a 	strd	r0, r1, [sp, #168]	; 0xa8
    a14e:	e9cd 012c 	strd	r0, r1, [sp, #176]	; 0xb0
    a152:	e9cd 012e 	strd	r0, r1, [sp, #184]	; 0xb8
    a156:	f992 1012 	ldrsb.w	r1, [r2, #18]
    a15a:	f8bd 00a4 	ldrh.w	r0, [sp, #164]	; 0xa4
    a15e:	f992 c009 	ldrsb.w	ip, [r2, #9]
    a162:	f8bd 908e 	ldrh.w	r9, [sp, #142]	; 0x8e
    a166:	f8cd c04c 	str.w	ip, [sp, #76]	; 0x4c
    a16a:	fb0a 0101 	mla	r1, sl, r1, r0
    a16e:	f8bd c08c 	ldrh.w	ip, [sp, #140]	; 0x8c
    a172:	f997 a013 	ldrsb.w	sl, [r7, #19]
    a176:	f8bd 6084 	ldrh.w	r6, [sp, #132]	; 0x84
    a17a:	f8bd 4086 	ldrh.w	r4, [sp, #134]	; 0x86
    a17e:	f992 3010 	ldrsb.w	r3, [r2, #16]
    a182:	f992 5011 	ldrsb.w	r5, [r2, #17]
    a186:	f8ad c030 	strh.w	ip, [sp, #48]	; 0x30
    a18a:	f8ad 903c 	strh.w	r9, [sp, #60]	; 0x3c
    a18e:	f997 c011 	ldrsb.w	ip, [r7, #17]
    a192:	f997 9010 	ldrsb.w	r9, [r7, #16]
    a196:	f8ad 10a4 	strh.w	r1, [sp, #164]	; 0xa4
    a19a:	f992 1013 	ldrsb.w	r1, [r2, #19]
    a19e:	f8bd 00a6 	ldrh.w	r0, [sp, #166]	; 0xa6
    a1a2:	fb0a 0101 	mla	r1, sl, r1, r0
    a1a6:	f992 a00a 	ldrsb.w	sl, [r2, #10]
    a1aa:	f8cd a054 	str.w	sl, [sp, #84]	; 0x54
    a1ae:	f997 a00b 	ldrsb.w	sl, [r7, #11]
    a1b2:	f8cd a058 	str.w	sl, [sp, #88]	; 0x58
    a1b6:	f04f 0a00 	mov.w	sl, #0
    a1ba:	fb08 ae0e 	mla	lr, r8, lr, sl
    a1be:	f8cd e000 	str.w	lr, [sp]
    a1c2:	46d6      	mov	lr, sl
    a1c4:	fb09 e303 	mla	r3, r9, r3, lr
    a1c8:	f8ad 10a6 	strh.w	r1, [sp, #166]	; 0xa6
    a1cc:	f997 e00c 	ldrsb.w	lr, [r7, #12]
    a1d0:	f992 1014 	ldrsb.w	r1, [r2, #20]
    a1d4:	9117      	str	r1, [sp, #92]	; 0x5c
    a1d6:	f8cd e060 	str.w	lr, [sp, #96]	; 0x60
    a1da:	f992 e00c 	ldrsb.w	lr, [r2, #12]
    a1de:	f997 0014 	ldrsb.w	r0, [r7, #20]
    a1e2:	9917      	ldr	r1, [sp, #92]	; 0x5c
    a1e4:	f8cd e064 	str.w	lr, [sp, #100]	; 0x64
    a1e8:	f997 e00d 	ldrsb.w	lr, [r7, #13]
    a1ec:	f8ad 30a0 	strh.w	r3, [sp, #160]	; 0xa0
    a1f0:	f8cd e068 	str.w	lr, [sp, #104]	; 0x68
    a1f4:	4653      	mov	r3, sl
    a1f6:	f992 e00d 	ldrsb.w	lr, [r2, #13]
    a1fa:	f8cd e06c 	str.w	lr, [sp, #108]	; 0x6c
    a1fe:	fb00 3001 	mla	r0, r0, r1, r3
    a202:	f997 e00e 	ldrsb.w	lr, [r7, #14]
    a206:	9903      	ldr	r1, [sp, #12]
    a208:	f8cd e070 	str.w	lr, [sp, #112]	; 0x70
    a20c:	f992 e00e 	ldrsb.w	lr, [r2, #14]
    a210:	f8cd e074 	str.w	lr, [sp, #116]	; 0x74
    a214:	fb0b 6e01 	mla	lr, fp, r1, r6
    a218:	9e04      	ldr	r6, [sp, #16]
    a21a:	9905      	ldr	r1, [sp, #20]
    a21c:	f8cd e00c 	str.w	lr, [sp, #12]
    a220:	fb06 4e01 	mla	lr, r6, r1, r4
    a224:	fb0c 3505 	mla	r5, ip, r5, r3
    a228:	f8cd e010 	str.w	lr, [sp, #16]
    a22c:	f997 e00f 	ldrsb.w	lr, [r7, #15]
    a230:	f8ad 50a2 	strh.w	r5, [sp, #162]	; 0xa2
    a234:	f992 500b 	ldrsb.w	r5, [r2, #11]
    a238:	f8cd e078 	str.w	lr, [sp, #120]	; 0x78
    a23c:	f8bd 3096 	ldrh.w	r3, [sp, #150]	; 0x96
    a240:	f8ad 305c 	strh.w	r3, [sp, #92]	; 0x5c
    a244:	9b0b      	ldr	r3, [sp, #44]	; 0x2c
    a246:	f992 e00f 	ldrsb.w	lr, [r2, #15]
    a24a:	9c0a      	ldr	r4, [sp, #40]	; 0x28
    a24c:	f8cd e07c 	str.w	lr, [sp, #124]	; 0x7c
    a250:	469e      	mov	lr, r3
    a252:	f8bd 3030 	ldrh.w	r3, [sp, #48]	; 0x30
    a256:	f8ad 00a8 	strh.w	r0, [sp, #168]	; 0xa8
    a25a:	fb04 3c0e 	mla	ip, r4, lr, r3
    a25e:	9b0e      	ldr	r3, [sp, #56]	; 0x38
    a260:	9c0d      	ldr	r4, [sp, #52]	; 0x34
    a262:	f8cd c014 	str.w	ip, [sp, #20]
    a266:	469c      	mov	ip, r3
    a268:	f8bd 303c 	ldrh.w	r3, [sp, #60]	; 0x3c
    a26c:	f8bd 00ac 	ldrh.w	r0, [sp, #172]	; 0xac
    a270:	f997 9015 	ldrsb.w	r9, [r7, #21]
    a274:	f997 8018 	ldrsb.w	r8, [r7, #24]
    a278:	f997 e019 	ldrsb.w	lr, [r7, #25]
    a27c:	f8bd 1094 	ldrh.w	r1, [sp, #148]	; 0x94
    a280:	f8bd b09c 	ldrh.w	fp, [sp, #156]	; 0x9c
    a284:	f8bd a09e 	ldrh.w	sl, [sp, #158]	; 0x9e
    a288:	fb04 3c0c 	mla	ip, r4, ip, r3
    a28c:	f8cd c028 	str.w	ip, [sp, #40]	; 0x28
    a290:	f992 3016 	ldrsb.w	r3, [r2, #22]
    a294:	f997 c016 	ldrsb.w	ip, [r7, #22]
    a298:	f992 4015 	ldrsb.w	r4, [r2, #21]
    a29c:	fb0c 0303 	mla	r3, ip, r3, r0
    a2a0:	f8ad 30ac 	strh.w	r3, [sp, #172]	; 0xac
    a2a4:	f997 c017 	ldrsb.w	ip, [r7, #23]
    a2a8:	f8bd 00ae 	ldrh.w	r0, [sp, #174]	; 0xae
    a2ac:	f992 3017 	ldrsb.w	r3, [r2, #23]
    a2b0:	fb0c 0303 	mla	r3, ip, r3, r0
    a2b4:	f8ad 30ae 	strh.w	r3, [sp, #174]	; 0xae
    a2b8:	f992 3019 	ldrsb.w	r3, [r2, #25]
    a2bc:	f992 0018 	ldrsb.w	r0, [r2, #24]
    a2c0:	930f      	str	r3, [sp, #60]	; 0x3c
    a2c2:	f997 301a 	ldrsb.w	r3, [r7, #26]
    a2c6:	9e15      	ldr	r6, [sp, #84]	; 0x54
    a2c8:	46b4      	mov	ip, r6
    a2ca:	9e14      	ldr	r6, [sp, #80]	; 0x50
    a2cc:	fb06 160c 	mla	r6, r6, ip, r1
    a2d0:	960b      	str	r6, [sp, #44]	; 0x2c
    a2d2:	f8bd 105c 	ldrh.w	r1, [sp, #92]	; 0x5c
    a2d6:	9e16      	ldr	r6, [sp, #88]	; 0x58
    a2d8:	fb06 1c05 	mla	ip, r6, r5, r1
    a2dc:	f997 601c 	ldrsb.w	r6, [r7, #28]
    a2e0:	9614      	str	r6, [sp, #80]	; 0x50
    a2e2:	9e1d      	ldr	r6, [sp, #116]	; 0x74
    a2e4:	f8cd c030 	str.w	ip, [sp, #48]	; 0x30
    a2e8:	46b4      	mov	ip, r6
    a2ea:	9e1c      	ldr	r6, [sp, #112]	; 0x70
    a2ec:	f992 101a 	ldrsb.w	r1, [r2, #26]
    a2f0:	f8bd 50b4 	ldrh.w	r5, [sp, #180]	; 0xb4
    a2f4:	fb06 bc0c 	mla	ip, r6, ip, fp
    a2f8:	9e1f      	ldr	r6, [sp, #124]	; 0x7c
    a2fa:	f8cd c034 	str.w	ip, [sp, #52]	; 0x34
    a2fe:	46b4      	mov	ip, r6
    a300:	9e1e      	ldr	r6, [sp, #120]	; 0x78
    a302:	fb03 5301 	mla	r3, r3, r1, r5
    a306:	fb06 ac0c 	mla	ip, r6, ip, sl
    a30a:	f04f 0a00 	mov.w	sl, #0
    a30e:	f997 501b 	ldrsb.w	r5, [r7, #27]
    a312:	f8bd 10b6 	ldrh.w	r1, [sp, #182]	; 0xb6
    a316:	f8ad 30b4 	strh.w	r3, [sp, #180]	; 0xb4
    a31a:	fb09 a404 	mla	r4, r9, r4, sl
    a31e:	f992 301b 	ldrsb.w	r3, [r2, #27]
    a322:	f8ad 40aa 	strh.w	r4, [sp, #170]	; 0xaa
    a326:	4654      	mov	r4, sl
    a328:	fb05 1303 	mla	r3, r5, r3, r1
    a32c:	fb08 4000 	mla	r0, r8, r0, r4
    a330:	9c0f      	ldr	r4, [sp, #60]	; 0x3c
    a332:	f997 501d 	ldrsb.w	r5, [r7, #29]
    a336:	f992 101c 	ldrsb.w	r1, [r2, #28]
    a33a:	9e14      	ldr	r6, [sp, #80]	; 0x50
    a33c:	f8ad 30b6 	strh.w	r3, [sp, #182]	; 0xb6
    a340:	f8ad 00b0 	strh.w	r0, [sp, #176]	; 0xb0
    a344:	f992 301d 	ldrsb.w	r3, [r2, #29]
    a348:	f8cd c038 	str.w	ip, [sp, #56]	; 0x38
    a34c:	4650      	mov	r0, sl
    a34e:	fb0e 0c04 	mla	ip, lr, r4, r0
    a352:	4654      	mov	r4, sl
    a354:	fb06 4101 	mla	r1, r6, r1, r4
    a358:	fb05 0303 	mla	r3, r5, r3, r0
    a35c:	f8ad c0b2 	strh.w	ip, [sp, #178]	; 0xb2
    a360:	f997 c01e 	ldrsb.w	ip, [r7, #30]
    a364:	f8ad 10b8 	strh.w	r1, [sp, #184]	; 0xb8
    a368:	f8ad 30ba 	strh.w	r3, [sp, #186]	; 0xba
    a36c:	f8cd c03c 	str.w	ip, [sp, #60]	; 0x3c
    a370:	f992 3023 	ldrsb.w	r3, [r2, #35]	; 0x23
    a374:	f997 5023 	ldrsb.w	r5, [r7, #35]	; 0x23
    a378:	f997 b027 	ldrsb.w	fp, [r7, #39]	; 0x27
    a37c:	f992 1022 	ldrsb.w	r1, [r2, #34]	; 0x22
    a380:	f997 6022 	ldrsb.w	r6, [r7, #34]	; 0x22
    a384:	f997 9021 	ldrsb.w	r9, [r7, #33]	; 0x21
    a388:	f997 4026 	ldrsb.w	r4, [r7, #38]	; 0x26
    a38c:	f997 e024 	ldrsb.w	lr, [r7, #36]	; 0x24
    a390:	f992 c01e 	ldrsb.w	ip, [r2, #30]
    a394:	f992 0020 	ldrsb.w	r0, [r2, #32]
    a398:	f997 8020 	ldrsb.w	r8, [r7, #32]
    a39c:	f8cd c050 	str.w	ip, [sp, #80]	; 0x50
    a3a0:	fb15 f503 	smulbb	r5, r5, r3
    a3a4:	f992 3027 	ldrsb.w	r3, [r2, #39]	; 0x27
    a3a8:	f997 c025 	ldrsb.w	ip, [r7, #37]	; 0x25
    a3ac:	fb1b fb03 	smulbb	fp, fp, r3
    a3b0:	f992 3021 	ldrsb.w	r3, [r2, #33]	; 0x21
    a3b4:	fb16 f601 	smulbb	r6, r6, r1
    a3b8:	fb19 f903 	smulbb	r9, r9, r3
    a3bc:	f992 1026 	ldrsb.w	r1, [r2, #38]	; 0x26
    a3c0:	f992 3024 	ldrsb.w	r3, [r2, #36]	; 0x24
    a3c4:	fb14 f401 	smulbb	r4, r4, r1
    a3c8:	fb1e fe03 	smulbb	lr, lr, r3
    a3cc:	9900      	ldr	r1, [sp, #0]
    a3ce:	f992 3025 	ldrsb.w	r3, [r2, #37]	; 0x25
    a3d2:	fb18 f800 	smulbb	r8, r8, r0
    a3d6:	fa18 f881 	uxtah	r8, r8, r1
    a3da:	fb1c f103 	smulbb	r1, ip, r3
    a3de:	9115      	str	r1, [sp, #84]	; 0x54
    a3e0:	9903      	ldr	r1, [sp, #12]
    a3e2:	9b01      	ldr	r3, [sp, #4]
    a3e4:	f8ad 8080 	strh.w	r8, [sp, #128]	; 0x80
    a3e8:	fa16 f681 	uxtah	r6, r6, r1
    a3ec:	9904      	ldr	r1, [sp, #16]
    a3ee:	f8ad 6084 	strh.w	r6, [sp, #132]	; 0x84
    a3f2:	fa15 f581 	uxtah	r5, r5, r1
    a3f6:	9902      	ldr	r1, [sp, #8]
    a3f8:	f992 601f 	ldrsb.w	r6, [r2, #31]
    a3fc:	4650      	mov	r0, sl
    a3fe:	fb03 0001 	mla	r0, r3, r1, r0
    a402:	9000      	str	r0, [sp, #0]
    a404:	9807      	ldr	r0, [sp, #28]
    a406:	f8bd a0bc 	ldrh.w	sl, [sp, #188]	; 0xbc
    a40a:	4680      	mov	r8, r0
    a40c:	9806      	ldr	r0, [sp, #24]
    a40e:	f04f 0100 	mov.w	r1, #0
    a412:	460b      	mov	r3, r1
    a414:	fb00 1108 	mla	r1, r0, r8, r1
    a418:	f997 801f 	ldrsb.w	r8, [r7, #31]
    a41c:	f8ad 5086 	strh.w	r5, [sp, #134]	; 0x86
    a420:	9809      	ldr	r0, [sp, #36]	; 0x24
    a422:	f8bd 50be 	ldrh.w	r5, [sp, #190]	; 0xbe
    a426:	4684      	mov	ip, r0
    a428:	9808      	ldr	r0, [sp, #32]
    a42a:	fa1e fe81 	uxtah	lr, lr, r1
    a42e:	9915      	ldr	r1, [sp, #84]	; 0x54
    a430:	f8ad e088 	strh.w	lr, [sp, #136]	; 0x88
    a434:	fb00 330c 	mla	r3, r0, ip, r3
    a438:	9805      	ldr	r0, [sp, #20]
    a43a:	fa11 f383 	uxtah	r3, r1, r3
    a43e:	9914      	ldr	r1, [sp, #80]	; 0x50
    a440:	f8ad 308a 	strh.w	r3, [sp, #138]	; 0x8a
    a444:	fa14 f480 	uxtah	r4, r4, r0
    a448:	980a      	ldr	r0, [sp, #40]	; 0x28
    a44a:	f992 302e 	ldrsb.w	r3, [r2, #46]	; 0x2e
    a44e:	f8ad 408c 	strh.w	r4, [sp, #140]	; 0x8c
    a452:	468c      	mov	ip, r1
    a454:	990f      	ldr	r1, [sp, #60]	; 0x3c
    a456:	f997 402a 	ldrsb.w	r4, [r7, #42]	; 0x2a
    a45a:	9401      	str	r4, [sp, #4]
    a45c:	fa1b fb80 	uxtah	fp, fp, r0
    a460:	9800      	ldr	r0, [sp, #0]
    a462:	f997 4029 	ldrsb.w	r4, [r7, #41]	; 0x29
    a466:	f8ad b08e 	strh.w	fp, [sp, #142]	; 0x8e
    a46a:	fb01 aa0c 	mla	sl, r1, ip, sl
    a46e:	f997 c02b 	ldrsb.w	ip, [r7, #43]	; 0x2b
    a472:	f997 1028 	ldrsb.w	r1, [r7, #40]	; 0x28
    a476:	f8cd c008 	str.w	ip, [sp, #8]
    a47a:	fb08 5806 	mla	r8, r8, r6, r5
    a47e:	f997 c02e 	ldrsb.w	ip, [r7, #46]	; 0x2e
    a482:	f997 b02f 	ldrsb.w	fp, [r7, #47]	; 0x2f
    a486:	f992 602a 	ldrsb.w	r6, [r2, #42]	; 0x2a
    a48a:	f992 502f 	ldrsb.w	r5, [r2, #47]	; 0x2f
    a48e:	f8ad a0bc 	strh.w	sl, [sp, #188]	; 0xbc
    a492:	fa19 f080 	uxtah	r0, r9, r0
    a496:	f8ad 0082 	strh.w	r0, [sp, #130]	; 0x82
    a49a:	f8ad 80be 	strh.w	r8, [sp, #190]	; 0xbe
    a49e:	f992 0028 	ldrsb.w	r0, [r2, #40]	; 0x28
    a4a2:	9100      	str	r1, [sp, #0]
    a4a4:	f8cd c014 	str.w	ip, [sp, #20]
    a4a8:	f992 102b 	ldrsb.w	r1, [r2, #43]	; 0x2b
    a4ac:	9306      	str	r3, [sp, #24]
    a4ae:	f992 9029 	ldrsb.w	r9, [r2, #41]	; 0x29
    a4b2:	f997 c02c 	ldrsb.w	ip, [r7, #44]	; 0x2c
    a4b6:	f8cd c00c 	str.w	ip, [sp, #12]
    a4ba:	f997 c02d 	ldrsb.w	ip, [r7, #45]	; 0x2d
    a4be:	9f00      	ldr	r7, [sp, #0]
    a4c0:	f992 e02c 	ldrsb.w	lr, [r2, #44]	; 0x2c
    a4c4:	f8cd c010 	str.w	ip, [sp, #16]
    a4c8:	fb17 f000 	smulbb	r0, r7, r0
    a4cc:	9f11      	ldr	r7, [sp, #68]	; 0x44
    a4ce:	f992 c02d 	ldrsb.w	ip, [r2, #45]	; 0x2d
    a4d2:	9000      	str	r0, [sp, #0]
    a4d4:	463b      	mov	r3, r7
    a4d6:	9f10      	ldr	r7, [sp, #64]	; 0x40
    a4d8:	9801      	ldr	r0, [sp, #4]
    a4da:	f04f 0200 	mov.w	r2, #0
    a4de:	4692      	mov	sl, r2
    a4e0:	4690      	mov	r8, r2
    a4e2:	fb07 2203 	mla	r2, r7, r3, r2
    a4e6:	9f13      	ldr	r7, [sp, #76]	; 0x4c
    a4e8:	fb10 f606 	smulbb	r6, r0, r6
    a4ec:	fb1b f505 	smulbb	r5, fp, r5
    a4f0:	9802      	ldr	r0, [sp, #8]
    a4f2:	46bb      	mov	fp, r7
    a4f4:	9f12      	ldr	r7, [sp, #72]	; 0x48
    a4f6:	fb10 f101 	smulbb	r1, r0, r1
    a4fa:	fb07 aa0b 	mla	sl, r7, fp, sl
    a4fe:	e9dd 0305 	ldrd	r0, r3, [sp, #20]
    a502:	9f19      	ldr	r7, [sp, #100]	; 0x64
    a504:	fb14 f409 	smulbb	r4, r4, r9
    a508:	46b9      	mov	r9, r7
    a50a:	9f18      	ldr	r7, [sp, #96]	; 0x60
    a50c:	fb10 f303 	smulbb	r3, r0, r3
    a510:	980b      	ldr	r0, [sp, #44]	; 0x2c
    a512:	fb07 8809 	mla	r8, r7, r9, r8
    a516:	9f03      	ldr	r7, [sp, #12]
    a518:	fa16 f680 	uxtah	r6, r6, r0
    a51c:	9800      	ldr	r0, [sp, #0]
    a51e:	f8ad 6094 	strh.w	r6, [sp, #148]	; 0x94
    a522:	fb17 fe0e 	smulbb	lr, r7, lr
    a526:	9f04      	ldr	r7, [sp, #16]
    a528:	fa10 f282 	uxtah	r2, r0, r2
    a52c:	980c      	ldr	r0, [sp, #48]	; 0x30
    a52e:	f8ad 2090 	strh.w	r2, [sp, #144]	; 0x90
    a532:	fb17 fc0c 	smulbb	ip, r7, ip
    a536:	9f1b      	ldr	r7, [sp, #108]	; 0x6c
    a538:	fa11 f180 	uxtah	r1, r1, r0
    a53c:	980d      	ldr	r0, [sp, #52]	; 0x34
    a53e:	f8ad 1096 	strh.w	r1, [sp, #150]	; 0x96
    a542:	46b9      	mov	r9, r7
    a544:	9f1a      	ldr	r7, [sp, #104]	; 0x68
    a546:	f04f 0b00 	mov.w	fp, #0
    a54a:	fa13 f380 	uxtah	r3, r3, r0
    a54e:	980e      	ldr	r0, [sp, #56]	; 0x38
    a550:	f8ad 309c 	strh.w	r3, [sp, #156]	; 0x9c
    a554:	fb07 b709 	mla	r7, r7, r9, fp
    a558:	fa15 f580 	uxtah	r5, r5, r0
    a55c:	fa14 f48a 	uxtah	r4, r4, sl
    a560:	fa1e f888 	uxtah	r8, lr, r8
    a564:	fa1c fc87 	uxtah	ip, ip, r7
    a568:	f8ad 509e 	strh.w	r5, [sp, #158]	; 0x9e
    a56c:	f10d 027e 	add.w	r2, sp, #126	; 0x7e
    a570:	f8ad 4092 	strh.w	r4, [sp, #146]	; 0x92
    a574:	f8ad 8098 	strh.w	r8, [sp, #152]	; 0x98
    a578:	f8ad c09a 	strh.w	ip, [sp, #154]	; 0x9a
    a57c:	f10d 01be 	add.w	r1, sp, #190	; 0xbe
    a580:	4658      	mov	r0, fp
    a582:	f832 3f02 	ldrh.w	r3, [r2, #2]!
    a586:	4418      	add	r0, r3
    a588:	428a      	cmp	r2, r1
    a58a:	b200      	sxth	r0, r0
    a58c:	d1f9      	bne.n	a582 <_Z15dot_microkernelILj48EaasET2_PKT0_PKT1_+0x4e6>
    a58e:	b031      	add	sp, #196	; 0xc4
    a590:	e8bd 8ff0 	ldmia.w	sp!, {r4, r5, r6, r7, r8, r9, sl, fp, pc}
