00008368 <_Z15dot_microkernelILj48EaalET2_PKT0_PKT1_>:
    8368:	b470      	push	{r4, r5, r6}
    836a:	f990 2001 	ldrsb.w	r2, [r0, #1]
    836e:	f991 4001 	ldrsb.w	r4, [r1, #1]
    8372:	f991 6000 	ldrsb.w	r6, [r1]
    8376:	f990 3000 	ldrsb.w	r3, [r0]
    837a:	f991 5002 	ldrsb.w	r5, [r1, #2]
    837e:	fb12 fc04 	smulbb	ip, r2, r4
    8382:	f990 4002 	ldrsb.w	r4, [r0, #2]
    8386:	f990 2003 	ldrsb.w	r2, [r0, #3]
    838a:	fb13 c306 	smlabb	r3, r3, r6, ip
    838e:	f991 6003 	ldrsb.w	r6, [r1, #3]
    8392:	fb14 3c05 	smlabb	ip, r4, r5, r3
    8396:	f990 3004 	ldrsb.w	r3, [r0, #4]
    839a:	f991 5004 	ldrsb.w	r5, [r1, #4]
    839e:	f990 4005 	ldrsb.w	r4, [r0, #5]
    83a2:	fb12 c206 	smlabb	r2, r2, r6, ip
    83a6:	f991 6005 	ldrsb.w	r6, [r1, #5]
    83aa:	fb13 2c05 	smlabb	ip, r3, r5, r2
    83ae:	f990 2006 	ldrsb.w	r2, [r0, #6]
    83b2:	f991 5006 	ldrsb.w	r5, [r1, #6]
    83b6:	f990 3007 	ldrsb.w	r3, [r0, #7]
    83ba:	fb14 c406 	smlabb	r4, r4, r6, ip
    83be:	f991 6007 	ldrsb.w	r6, [r1, #7]
    83c2:	fb12 4c05 	smlabb	ip, r2, r5, r4
    83c6:	f990 4008 	ldrsb.w	r4, [r0, #8]
    83ca:	f991 5008 	ldrsb.w	r5, [r1, #8]
    83ce:	f990 2009 	ldrsb.w	r2, [r0, #9]
    83d2:	fb13 c306 	smlabb	r3, r3, r6, ip
    83d6:	f991 6009 	ldrsb.w	r6, [r1, #9]
    83da:	fb14 3c05 	smlabb	ip, r4, r5, r3
    83de:	f990 300a 	ldrsb.w	r3, [r0, #10]
    83e2:	f991 500a 	ldrsb.w	r5, [r1, #10]
    83e6:	f990 400b 	ldrsb.w	r4, [r0, #11]
    83ea:	fb12 c206 	smlabb	r2, r2, r6, ip
    83ee:	f991 600b 	ldrsb.w	r6, [r1, #11]
    83f2:	fb13 2c05 	smlabb	ip, r3, r5, r2
    83f6:	f990 200c 	ldrsb.w	r2, [r0, #12]
    83fa:	f991 500c 	ldrsb.w	r5, [r1, #12]
    83fe:	f990 300d 	ldrsb.w	r3, [r0, #13]
    8402:	fb14 c406 	smlabb	r4, r4, r6, ip
    8406:	f991 600d 	ldrsb.w	r6, [r1, #13]
    840a:	fb12 4c05 	smlabb	ip, r2, r5, r4
    840e:	f990 400e 	ldrsb.w	r4, [r0, #14]
    8412:	f991 500e 	ldrsb.w	r5, [r1, #14]
    8416:	f990 200f 	ldrsb.w	r2, [r0, #15]
    841a:	fb13 c306 	smlabb	r3, r3, r6, ip
    841e:	f991 600f 	ldrsb.w	r6, [r1, #15]
    8422:	fb14 3c05 	smlabb	ip, r4, r5, r3
    8426:	f990 3010 	ldrsb.w	r3, [r0, #16]
    842a:	f991 5010 	ldrsb.w	r5, [r1, #16]
    842e:	f990 4011 	ldrsb.w	r4, [r0, #17]
    8432:	fb12 c206 	smlabb	r2, r2, r6, ip
    8436:	f991 6011 	ldrsb.w	r6, [r1, #17]
    843a:	fb13 2c05 	smlabb	ip, r3, r5, r2
    843e:	f990 2012 	ldrsb.w	r2, [r0, #18]
    8442:	f991 5012 	ldrsb.w	r5, [r1, #18]
    8446:	f990 3013 	ldrsb.w	r3, [r0, #19]
    844a:	fb14 c406 	smlabb	r4, r4, r6, ip
    844e:	f991 6013 	ldrsb.w	r6, [r1, #19]
    8452:	fb12 4c05 	smlabb	ip, r2, r5, r4
    8456:	f990 4014 	ldrsb.w	r4, [r0, #20]
    845a:	f991 5014 	ldrsb.w	r5, [r1, #20]
    845e:	f990 2015 	ldrsb.w	r2, [r0, #21]
    8462:	fb13 c306 	smlabb	r3, r3, r6, ip
    8466:	f991 6015 	ldrsb.w	r6, [r1, #21]
    846a:	fb14 3c05 	smlabb	ip, r4, r5, r3
    846e:	f990 3016 	ldrsb.w	r3, [r0, #22]
    8472:	f991 5016 	ldrsb.w	r5, [r1, #22]
    8476:	f990 4017 	ldrsb.w	r4, [r0, #23]
    847a:	fb12 c206 	smlabb	r2, r2, r6, ip
    847e:	f991 6017 	ldrsb.w	r6, [r1, #23]
    8482:	fb13 2c05 	smlabb	ip, r3, r5, r2
    8486:	f990 2018 	ldrsb.w	r2, [r0, #24]
    848a:	f991 5018 	ldrsb.w	r5, [r1, #24]
    848e:	f990 3019 	ldrsb.w	r3, [r0, #25]
    8492:	fb14 c406 	smlabb	r4, r4, r6, ip
    8496:	f991 6019 	ldrsb.w	r6, [r1, #25]
    849a:	fb12 4c05 	smlabb	ip, r2, r5, r4
    849e:	f990 401a 	ldrsb.w	r4, [r0, #26]
    84a2:	f991 501a 	ldrsb.w	r5, [r1, #26]
    84a6:	f990 201b 	ldrsb.w	r2, [r0, #27]
    84aa:	fb13 c306 	smlabb	r3, r3, r6, ip
    84ae:	f991 601b 	ldrsb.w	r6, [r1, #27]
    84b2:	fb14 3c05 	smlabb	ip, r4, r5, r3
    84b6:	f990 301c 	ldrsb.w	r3, [r0, #28]
    84ba:	f991 501c 	ldrsb.w	r5, [r1, #28]
    84be:	f990 401d 	ldrsb.w	r4, [r0, #29]
    84c2:	fb12 c206 	smlabb	r2, r2, r6, ip
    84c6:	f991 601d 	ldrsb.w	r6, [r1, #29]
    84ca:	fb13 2c05 	smlabb	ip, r3, r5, r2
    84ce:	f990 201e 	ldrsb.w	r2, [r0, #30]
    84d2:	f991 501e 	ldrsb.w	r5, [r1, #30]
    84d6:	f990 301f 	ldrsb.w	r3, [r0, #31]
    84da:	fb14 c406 	smlabb	r4, r4, r6, ip
    84de:	f991 601f 	ldrsb.w	r6, [r1, #31]
    84e2:	fb12 4c05 	smlabb	ip, r2, r5, r4
    84e6:	f990 4020 	ldrsb.w	r4, [r0, #32]
    84ea:	f991 5020 	ldrsb.w	r5, [r1, #32]
    84ee:	f990 2021 	ldrsb.w	r2, [r0, #33]	; 0x21
    84f2:	fb13 c306 	smlabb	r3, r3, r6, ip
    84f6:	f991 6021 	ldrsb.w	r6, [r1, #33]	; 0x21
    84fa:	fb14 3c05 	smlabb	ip, r4, r5, r3
    84fe:	f990 3022 	ldrsb.w	r3, [r0, #34]	; 0x22
    8502:	f991 5022 	ldrsb.w	r5, [r1, #34]	; 0x22
    8506:	f990 4023 	ldrsb.w	r4, [r0, #35]	; 0x23
    850a:	fb12 c206 	smlabb	r2, r2, r6, ip
    850e:	f991 6023 	ldrsb.w	r6, [r1, #35]	; 0x23
    8512:	fb13 2c05 	smlabb	ip, r3, r5, r2
    8516:	f990 2024 	ldrsb.w	r2, [r0, #36]	; 0x24
    851a:	f991 5024 	ldrsb.w	r5, [r1, #36]	; 0x24
    851e:	f990 3025 	ldrsb.w	r3, [r0, #37]	; 0x25
    8522:	fb14 c406 	smlabb	r4, r4, r6, ip
    8526:	f991 6025 	ldrsb.w	r6, [r1, #37]	; 0x25
    852a:	fb12 4c05 	smlabb	ip, r2, r5, r4
    852e:	f990 4026 	ldrsb.w	r4, [r0, #38]	; 0x26
    8532:	f991 5026 	ldrsb.w	r5, [r1, #38]	; 0x26
    8536:	f990 2027 	ldrsb.w	r2, [r0, #39]	; 0x27
    853a:	fb13 c306 	smlabb	r3, r3, r6, ip
    853e:	f991 6027 	ldrsb.w	r6, [r1, #39]	; 0x27
    8542:	fb14 3c05 	smlabb	ip, r4, r5, r3
    8546:	f990 3028 	ldrsb.w	r3, [r0, #40]	; 0x28
    854a:	f991 5028 	ldrsb.w	r5, [r1, #40]	; 0x28
    854e:	f990 4029 	ldrsb.w	r4, [r0, #41]	; 0x29
    8552:	fb12 c206 	smlabb	r2, r2, r6, ip
    8556:	f991 6029 	ldrsb.w	r6, [r1, #41]	; 0x29
    855a:	fb13 2c05 	smlabb	ip, r3, r5, r2
    855e:	f990 202a 	ldrsb.w	r2, [r0, #42]	; 0x2a
    8562:	f991 502a 	ldrsb.w	r5, [r1, #42]	; 0x2a
    8566:	f990 302b 	ldrsb.w	r3, [r0, #43]	; 0x2b
    856a:	fb14 c406 	smlabb	r4, r4, r6, ip
    856e:	f991 602b 	ldrsb.w	r6, [r1, #43]	; 0x2b
    8572:	fb12 4c05 	smlabb	ip, r2, r5, r4
    8576:	f990 402c 	ldrsb.w	r4, [r0, #44]	; 0x2c
    857a:	f991 502c 	ldrsb.w	r5, [r1, #44]	; 0x2c
    857e:	f990 202d 	ldrsb.w	r2, [r0, #45]	; 0x2d
    8582:	fb13 c306 	smlabb	r3, r3, r6, ip
    8586:	f991 602d 	ldrsb.w	r6, [r1, #45]	; 0x2d
    858a:	fb14 3c05 	smlabb	ip, r4, r5, r3
    858e:	f990 402e 	ldrsb.w	r4, [r0, #46]	; 0x2e
    8592:	f991 502e 	ldrsb.w	r5, [r1, #46]	; 0x2e
    8596:	f990 002f 	ldrsb.w	r0, [r0, #47]	; 0x2f
    859a:	f991 102f 	ldrsb.w	r1, [r1, #47]	; 0x2f
    859e:	fb12 c306 	smlabb	r3, r2, r6, ip
    85a2:	fb14 3205 	smlabb	r2, r4, r5, r3
    85a6:	fb00 2001 	mla	r0, r0, r1, r2
    85aa:	bc70      	pop	{r4, r5, r6}
    85ac:	4770      	bx	lr
    85ae:	bf00      	nop
