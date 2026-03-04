#include "hal.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "simpleserial.h"

#define WORD_SIZE 16
#define ROUNDS 32
#define rol(x,r) (((x)<<(r)) | ((x)>>(16-(r))))
#define ror(x,r) (((x)>>(r)) | ((x)<<(16-(r))))

uint16_t key[4] = {0x00};
uint16_t ct[2] = {0x00};

// 轮密钥数组
uint16_t round_keys[ROUNDS];

// Simon 的常量 z 序列
const uint64_t z0 = 0b11111010001001010110000111001101111101000100101011000011100110;

// 单轮加密函数
void enc_one_round(uint16_t *left, uint16_t *right, uint16_t k) {
    uint16_t temp = *left;
    uint16_t f_result = (rol(temp, 1) & rol(temp, 8)) ^ rol(temp, 2);
//     trigger_high();
    *left = *right ^ f_result ^ k;
//     trigger_low();
    *right = temp;
}

// 密钥扩展
void key_schedule(uint16_t *key) {
    uint16_t c = 0xFFFC; 
    round_keys[0] = key[0];
    round_keys[1] = key[1];
    round_keys[2] = key[2];
    round_keys[3] = key[3];
    
    for (int i = 4; i < ROUNDS; i++) {
        uint16_t tmp = ror(round_keys[i-1], 3);
        // Simon 32/64 m=4 的密钥扩展核心逻辑
        tmp ^= round_keys[i-3];
        tmp ^= ror(tmp, 1);
        
        uint8_t z_bit = (z0 >> ((i - 4) % 62)) & 1;
        
        // 注意：去掉 ~ 号，直接使用 c
        round_keys[i] = round_keys[i-4] ^ tmp ^ z_bit ^ c;
    }
}

// 加密主函数
// void encrypt(uint16_t *pt, uint16_t *ct, uint16_t *key) {
//     // 生成所有轮密钥
//     key_schedule(key);
    
//     uint16_t left = pt[1];
//     uint16_t right = pt[0];
    
    
//     // 执行 32 轮加密
//     for (int i = 0; i < ROUNDS; i++) {
//         trigger_high();
//         enc_one_round(&left, &right, round_keys[i]);
//         trigger_low();
//     }
    
//     ct[0] = left;
//     ct[1] = right;
// }

// void encrypt(uint16_t *pt, uint16_t *ct, uint16_t *key) {
//     // 生成所有轮密钥
//     key_schedule(key);
    
//     uint16_t left = pt[1];
//     uint16_t right = pt[0];
    
//     trigger_high(); // <<<--- 在加密开始时拉高
//     // 执行 32 轮加密
//     for (int i = 0; i < ROUNDS; i++) {
//         enc_one_round(&left, &right, round_keys[i]);
//     }
//     trigger_low();  // <<<--- 在加密结束时拉低
    
//     ct[0] = left;
//     ct[1] = right;
// }


// 假设您将 enc_one_round 恢复到不带触发的版本，并将触发逻辑移入 encrypt
void encrypt(uint16_t *pt, uint16_t *ct, uint16_t *key) {
    key_schedule(key);
    uint16_t left = pt[1];
    uint16_t right = pt[0];
    
    for (int i = 0; i < ROUNDS; i++) {
        if (i ==0) {
            trigger_high(); // 仅在第一轮开始时拉高
        }
        
        // 确保 enc_one_round 不包含触发器
        enc_one_round(&left, &right, round_keys[i]);
        
        if (i == 0) {
            trigger_low();  // 仅在第一轮结束时拉低
        }
    }
    // ...
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
    
//     trigger_high();
    encrypt(pt16, ct, key);
//     trigger_low(); 
    
    uint8_t ct_bytes[4];
    ct_bytes[0] = (uint8_t)(ct[0] & 0xFF);
    ct_bytes[1] = (uint8_t)(ct[0] >> 8); 
    ct_bytes[2] = (uint8_t)(ct[1] & 0xFF); 
    ct_bytes[3] = (uint8_t)(ct[1] >> 8);
    
    simpleserial_put('r', 4, ct_bytes);
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