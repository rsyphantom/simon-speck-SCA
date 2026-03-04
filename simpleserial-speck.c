#include "hal.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "simpleserial.h"

#define WORD_SIZE 16
#define ALPHA 7
#define BETA 2
#define ROUNDS 22
#define rol(x,r) (((x)<<(r)) | (x>>(16-(r))))
#define ror(x,r) (((x)>>(r)) | ((x)<<(16-(r))))

uint16_t key[4]={0x00};
uint16_t ct[2]={0x00};

// 轮密钥数组
uint16_t round_keys[ROUNDS];

// 单轮加密
void enc_one_round(uint16_t *c1, uint16_t *c0, uint16_t k) {
    *c1 = ror(*c1, ALPHA);
    *c1 = (*c1 + *c0);
    *c1 = *c1 ^ k;
    *c0 = rol(*c0, BETA);
    *c0 = *c0 ^ *c1;
}
void key_schedule_round(uint16_t *l_val, uint16_t k_val, int i) {
    uint16_t tmp_l = ror(*l_val, 7);
    tmp_l = (tmp_l + k_val);
    tmp_l ^= i;
    uint16_t tmp_k = rol(k_val, 2);
    tmp_k ^= tmp_l;
    *l_val = tmp_l;
    round_keys[i+1] = tmp_k;
}

// 加密主函数
void encrypt(uint16_t *pt, uint16_t *ct, uint16_t *key) {
    
    uint16_t l[3];

    // 初始轮密钥和 L 值
    round_keys[0] = key[0];  // k0
    l[0] = key[1];           // l0
    l[1] = key[2];           // l1
    l[2] = key[3];  
    
    uint16_t x = pt[0];
    uint16_t y = pt[1];

    for (int i = 0; i < ROUNDS ; i++) {
        uint16_t tmp_l = l[i % 3];
        uint16_t tmp_k = round_keys[i];

        if (i ==0) {
            trigger_high(); // 仅在第一轮开始时拉高
        }
        
        // 确保 enc_one_round 不包含触发器
        enc_one_round(&y, &x, round_keys[i]);
        
        if (i == 0) {
            trigger_low();  // 仅在第一轮结束时拉低
        }
        
        // 对应 key_schedule_round 中的密钥扩展逻辑
        key_schedule_round(&l[i % 3], tmp_k, i);
    }

    ct[0] = x;  // right half
    ct[1] = y;  // left half
}


uint8_t get_key(uint8_t* k, uint8_t len)
{
   key[0] = (k[1] << 8) | k[0];  // 将前 2 字节转换为 16 位
   key[1] = (k[3] << 8) | k[2];
   key[2] = (k[5] << 8) | k[4];
   key[3] = (k[7] << 8) | k[6];
   return 0x00;
}

uint8_t get_pt(uint8_t* pt, uint8_t len)
{
   uint16_t pt16[2];  // 16 位输入数据

   pt16[0] = (pt[1] << 8) | pt[0];  // 将前 2 字节转换为 16 位
   pt16[1] = (pt[3] << 8) | pt[2];  // 将后 2 字节转换为 16 位

	trigger_high();
   
   encrypt(pt16, ct, key);

	trigger_low();
    
   uint8_t ct_bytes[4];
   ct_bytes[0] = (uint8_t)(ct[0] & 0xFF);
   ct_bytes[1] = (uint8_t)(ct[0] >> 8); 
   ct_bytes[2] = (uint8_t)(ct[1] & 0xFF); 
   ct_bytes[3] = (uint8_t)(ct[1] >> 8);

	simpleserial_put('r',4, ct_bytes);
	return 0x00;
}
int main(void)
{
   platform_init();
	init_uart();
	trigger_setup();

	simpleserial_init();

	simpleserial_addcmd('k', 8, get_key);
	simpleserial_addcmd('p', 4, get_pt);

	while(1)
		simpleserial_get();
}