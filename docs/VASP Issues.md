# Attack Name Discripency Issues:
- ***RandomPositionOffset* attack is implemented as  *RandomOffset***
- ***ConstantPositionOffset* attack is implemented as *ConstantOffset***

## Temp Workaround:
- During data prerprocessing step, rename *RandomOffset* or *ConstantOffset* attacks as *RandomPositionOffset*, and *ConstantPositionOffset* attacks, respectively.  


# Implementatation Issues:
- ***RandomAccelearation* (AttackID: 58) is creating -nan value in the field of rv_accel**
- Example of some bsms generated under *RandomAccelearation* attack:

```
rv_id	hv_id	target_id	msg_generation_time	msg_rcv_time	rv_msg_count	rv_wsm_data	rv_pos_x	rv_pos_y	rv_pos_z	rv_speed	rv_accel	rv_heading	rv_yaw_rate	rv_length	rv_width	rv_height	hv_msg_count	hv_wsm_data	hv_pos_x	hv_pos_y	hv_pos_z	hv_speed	hv_accel	hv_heading	hv_length	hv_width	hv_height	attack_type	eebl_warn	ima_warn
150	54	-1	0.11424	0.114346932163	0	genuine	2005.4	3066.98	0	0	0	-0.598108	-1	5	1.8	1.5	0	genuine	2032.21	3345.15	0	0	0	-0.0703826	5	1.8	1.5	Genuine	0	0
300	462	-1	0.118081	0.118194408253	0	genuine	1243.26	906.065	0	0	-nan	-0.822644	-1	5	1.8	1.5	0	genuine	1349.38	845.093	0	0	0	0.504847	5	1.8	1.5	RandomAcceleration	0	0
300	282	-1	0.118081	0.118194506679	0	genuine	1243.26	906.065	0	0	-nan	-0.822644	-1	5	1.8	1.5	0	genuine	1332.07	1029.29	0	0	0	-0.98571	5	1.8	1.5	RandomAcceleration	0	0
300	492	-1	0.118081	0.118195148403	0	genuine	1243.26	906.065	0	0	-nan	-0.822644	-1	5	1.8	1.5	0	genuine	1367.53	1227.14	0	0	0	-0.816614	5	1.8	1.5	RandomAcceleration	0	0
```

- ***RandomSpeed* (AttackID: 64) is creating -nan value in the field of rv_speed**
- Example of some bsms generated under *RandomSpeed* attack:
```
RandomSpeed is creating -nan value if the field of rv_speed
rv_id	hv_id	target_id	msg_generation_time	msg_rcv_time	rv_msg_count	rv_wsm_data	rv_pos_x	rv_pos_y	rv_pos_z	rv_speed	rv_accel	rv_heading	rv_yaw_rate	rv_length	rv_width	rv_height	hv_msg_count	hv_wsm_data	hv_pos_x	hv_pos_y	hv_pos_z	hv_speed	hv_accel	hv_heading	hv_length	hv_width	hv_height	attack_type	eebl_warn	ima_warn
54	228	-1	0.112786	0.11290461481	0	genuine	2032.21	3345.15	0	-nan	0	-0.0703826	-1	5	1.8	1.5	0	genuine	1899.78	2879.51	0	0	0	-0.888649	5	1.8	1.5	RandomSpeed	0	0
492	300	-1	0.112821	0.112930148403	0	genuine	1367.53	1227.14	0	0	0	-0.816614	-1	5	1.8	1.5	0	genuine	1243.26	906.065	0	0	0	-0.822644	5	1.8	1.5	Genuine	0	0
492	462	-1	0.112821	0.112930275798	0	genuine	1367.53	1227.14	0	0	0	-0.816614	-1	5	1.8	1.5	0	genuine	1349.38	845.093	0	0	0	0.504847	5	1.8	1.5	Genuine	0	0
462	300	-1	0.117939	0.118046684051	0	genuine	1349.38	845.093	0	0	0	0.504847	-1	5	1.8	1.5	0	genuine	1243.26	906.065	0	0	0	-0.822644	5	1.8	1.5	Genuine	0	0
462	492	-1	0.117939	0.118047551596	0	genuine	1349.38	845.093	0	0	0	0.504847	-1	5	1.8	1.5	1	genuine	1367.53	1227.14	0	0	0	-0.816614	5	1.8	1.5	Genuine	0	0
```

## Temp Workaround:
- During data prerprocessing step, replace the -nan values with randomly generated accell or speed during the RandomAccelearation and RandomSpeed attacks. Consider the range of values based on the min and max of the benign vehicles.


