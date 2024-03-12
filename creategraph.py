import os
import torch
import math


def cos_Loss(predict, truth, N):
    neijitensor = torch.mul(predict, truth)
    neijitensor = torch.sum(neijitensor, 1)
    t = predict * predict
    p = torch.sqrt(torch.sum(t, 1))
    t = truth * truth
    p1 = torch.sqrt(torch.sum(t, 1))
    loss = torch.mean(1 - torch.div(torch.div(neijitensor, p), p1))
    return loss


def angle_cos_Loss(predict1, predict2, truth1, truth2, N):
    neijitensor1 = torch.mul(predict1, predict2)
    neijitensor1 = torch.sum(neijitensor1, 1)
    t = predict1 * predict1
    p = torch.sqrt(torch.sum(t, 1))
    t = predict2 * predict2
    p1 = torch.sqrt(torch.sum(t, 1))
    loss1 = 1 - torch.div(torch.div(neijitensor1, p), p1)
    neijitensor1 = torch.mul(truth1, truth2)
    neijitensor1 = torch.sum(neijitensor1, 1)
    t = truth1 * truth1
    p = torch.sqrt(torch.sum(t, 1))
    t = truth2 * truth2
    p1 = torch.sqrt(torch.sum(t, 1))
    loss2 = 1 - torch.div(torch.div(neijitensor1, p), p1)
    diff = (loss1 - loss2) ** 2
    return torch.mean(diff)


def creategraphT(predict, groundturth):
    # predict = predict.numpy()
    # print(predict.shape)
    # print(groundturth.shape)
    # ????
    left_eye_sum = 0
    right_eye_sum = 0
    mouth_sum = 0
    left_eye_sum_truth = 0
    right_eye_sum_truth = 0
    mouth_sum_truth = 0

    # average eye
    for i in range(36, 42):
        left_eye_sum += predict[:, :, i]
        # print(predict[:,:,i])
        left_eye_sum_truth += groundturth[:, :, i]
    # print(left_eye_sum)

    left_eye_mean = left_eye_sum / 6
    left_eye_mean_truth = left_eye_sum_truth / 6

    # print(left_eye_mean)
    for i in range(42, 48):
        right_eye_sum += predict[:, :, i]
        right_eye_sum_truth += groundturth[:, :, i]
    right_eye_mean = right_eye_sum / 6
    right_eye_mean_truth = right_eye_sum_truth / 6
    for i in range(48, 62):
        mouth_sum += predict[:, :, i]
        mouth_sum_truth += groundturth[:, :, i]
    mouth_mean = mouth_sum / 20
    mouth_mean_truth = mouth_sum_truth / 20

    total_1 = left_eye_mean - predict[:, :, 30]
    total_2 = right_eye_mean - predict[:, :, 30]
    total_3 = mouth_mean - predict[:, :, 30]
    total_4 = predict[:, :, 8] - predict[:, :, 30]

    total_1_truth = left_eye_mean_truth - groundturth[:, :, 30]
    total_2_truth = right_eye_mean_truth - groundturth[:, :, 30]
    total_3_truth = mouth_mean_truth - groundturth[:, :, 30]
    total_4_truth = groundturth[:, :, 8] - groundturth[:, :, 30]
    # loss_angle = cos_Loss(total_1, total_1_truth, N)
    # loss_angle += cos_Loss(total_2, total_2_truth, N)
    # loss_angle += cos_Loss(total_3, total_3_truth, N)
    # loss_angle += cos_Loss(total_4, total_4_truth, N)

    loss = 10 * torch.mean((total_1 - total_1_truth) ** 2)
    loss += 10 * torch.mean((total_2 - total_2_truth) ** 2)
    loss += 10 * torch.mean((total_3 - total_3_truth) ** 2)
    loss += 10 * torch.mean((total_4 - total_4_truth) ** 2)
    # loss_angle1 = 10*angle_cos_Loss(total_1,total_2,total_1_truth,total_2_truth,N)

    # ????
    # ??????
    # eye_local_reative_distance = predict[:,:,17] - predict[:,:,22]
    # eye_local_reative_distance_truth = groundturth[:,:,17] - groundturth[:,:,22]
    # loss+= torch.mean((eye_local_reative_distance-eye_local_reative_distance_truth)**2)

    eye_18 = predict[:, :, 17] - left_eye_mean
    eye_19 = predict[:, :, 18] - left_eye_mean
    eye_20 = predict[:, :, 19] - left_eye_mean
    eye_21 = predict[:, :, 20] - left_eye_mean
    eye_22 = predict[:, :, 21] - left_eye_mean

    eye_18_truth = groundturth[:, :, 17] - left_eye_mean_truth
    eye_19_truth = groundturth[:, :, 18] - left_eye_mean_truth
    eye_20_truth = groundturth[:, :, 19] - left_eye_mean_truth
    eye_21_truth = groundturth[:, :, 20] - left_eye_mean_truth
    eye_22_truth = groundturth[:, :, 21] - left_eye_mean_truth

    loss += torch.mean((eye_18 - eye_18_truth) ** 2)
    loss += torch.mean((eye_19 - eye_19_truth) ** 2)
    loss += torch.mean((eye_20 - eye_20_truth) ** 2)
    loss += torch.mean((eye_21 - eye_21_truth) ** 2)
    loss += torch.mean((eye_22 - eye_22_truth) ** 2)

    # loss_angle += cos_Loss(eye_18, eye_18_truth, N)
    # loss_angle += cos_Loss(eye_19, eye_19_truth, N)
    # loss_angle += cos_Loss(eye_20, eye_20_truth, N)
    # loss_angle += cos_Loss(eye_21, eye_21_truth, N)
    # loss_angle += cos_Loss(eye_22, eye_22_truth, N)

    eye_37 = predict[:, :, 36] - left_eye_mean
    eye_38 = predict[:, :, 37] - left_eye_mean
    eye_39 = predict[:, :, 38] - left_eye_mean
    eye_40 = predict[:, :, 39] - left_eye_mean
    eye_41 = predict[:, :, 40] - left_eye_mean
    eye_42 = predict[:, :, 41] - left_eye_mean

    eye_37_truth = groundturth[:, :, 36] - left_eye_mean_truth
    eye_38_truth = groundturth[:, :, 37] - left_eye_mean_truth
    eye_39_truth = groundturth[:, :, 38] - left_eye_mean_truth
    eye_40_truth = groundturth[:, :, 39] - left_eye_mean_truth
    eye_41_truth = groundturth[:, :, 40] - left_eye_mean_truth
    eye_42_truth = groundturth[:, :, 41] - left_eye_mean_truth

    loss += torch.mean((eye_37 - eye_37_truth) ** 2)
    loss += torch.mean((eye_38 - eye_38_truth) ** 2)
    loss += torch.mean((eye_39 - eye_39_truth) ** 2)
    loss += torch.mean((eye_40 - eye_40_truth) ** 2)
    loss += torch.mean((eye_41 - eye_41_truth) ** 2)
    loss += torch.mean((eye_42 - eye_42_truth) ** 2)

    # loss_angle += cos_Loss(eye_37, eye_37_truth, N)
    # loss_angle += cos_Loss(eye_38, eye_38_truth, N)
    # loss_angle += cos_Loss(eye_39, eye_39_truth, N)
    # loss_angle += cos_Loss(eye_40, eye_40_truth, N)
    # loss_angle += cos_Loss(eye_41, eye_41_truth, N)
    # loss_angle += cos_Loss(eye_42, eye_42_truth, N)

    # ??????

    noise_28 = predict[:, :, 27] - predict[:, :, 30]
    noise_29 = predict[:, :, 28] - predict[:, :, 30]
    noise_30 = predict[:, :, 29] - predict[:, :, 30]
    noise_32 = predict[:, :, 31] - predict[:, :, 30]
    noise_33 = predict[:, :, 32] - predict[:, :, 30]
    noise_34 = predict[:, :, 33] - predict[:, :, 30]
    noise_35 = predict[:, :, 34] - predict[:, :, 30]
    noise_36 = predict[:, :, 35] - predict[:, :, 30]

    noise_28_truth = groundturth[:, :, 27] - groundturth[:, :, 30]
    noise_29_truth = groundturth[:, :, 28] - groundturth[:, :, 30]
    noise_30_truth = groundturth[:, :, 29] - groundturth[:, :, 30]
    noise_32_truth = groundturth[:, :, 31] - groundturth[:, :, 30]
    noise_33_truth = groundturth[:, :, 32] - groundturth[:, :, 30]
    noise_34_truth = groundturth[:, :, 33] - groundturth[:, :, 30]
    noise_35_truth = groundturth[:, :, 34] - groundturth[:, :, 30]
    noise_36_truth = groundturth[:, :, 35] - groundturth[:, :, 30]

    loss += torch.mean((noise_28 - noise_28_truth) ** 2)
    loss += torch.mean((noise_29 - noise_29_truth) ** 2)
    loss += torch.mean((noise_30 - noise_30_truth) ** 2)
    loss += torch.mean((noise_36 - noise_36_truth) ** 2)
    loss += torch.mean((noise_32 - noise_32_truth) ** 2)
    loss += torch.mean((noise_33 - noise_33_truth) ** 2)
    loss += torch.mean((noise_34 - noise_34_truth) ** 2)
    loss += torch.mean((noise_35 - noise_35_truth) ** 2)

    # loss_angle += cos_Loss(noise_28, noise_28_truth, N)
    # loss_angle += cos_Loss(noise_29, noise_29_truth, N)
    # loss_angle += cos_Loss(noise_30, noise_30_truth, N)
    # loss_angle += cos_Loss(noise_32, noise_32_truth, N)
    # loss_angle += cos_Loss(noise_33, noise_33_truth, N)
    # loss_angle += cos_Loss(noise_34, noise_34_truth, N)
    # loss_angle += cos_Loss(noise_35, noise_35_truth, N)
    # loss_angle += cos_Loss(noise_28, noise_28_truth, N)
    # ??????

    mouth_49 = predict[:, :, 48] - mouth_mean
    mouth_50 = predict[:, :, 49] - mouth_mean
    mouth_51 = predict[:, :, 50] - mouth_mean
    mouth_52 = predict[:, :, 51] - mouth_mean
    mouth_53 = predict[:, :, 52] - mouth_mean
    mouth_54 = predict[:, :, 53] - mouth_mean
    mouth_55 = predict[:, :, 54] - mouth_mean
    mouth_56 = predict[:, :, 55] - mouth_mean
    mouth_57 = predict[:, :, 56] - mouth_mean
    mouth_58 = predict[:, :, 57] - mouth_mean
    mouth_59 = predict[:, :, 58] - mouth_mean
    mouth_60 = predict[:, :, 59] - mouth_mean
    mouth_61 = predict[:, :, 60] - mouth_mean
    mouth_62 = predict[:, :, 61] - mouth_mean
    # mouth_63 = predict[:,:,62] - mouth_mean
    # mouth_64 = predict[:,:,63] - mouth_mean
    # mouth_65 = predict[:,:,64] - mouth_mean
    # mouth_66 = predict[:,:,65] - mouth_mean
    # mouth_67 = predict[:,:,66] - mouth_mean
    # mouth_68 = predict[:,:,67] - mouth_mean

    mouth_49_truth = groundturth[:, :, 48] - mouth_mean_truth
    mouth_50_truth = groundturth[:, :, 49] - mouth_mean_truth
    mouth_51_truth = groundturth[:, :, 50] - mouth_mean_truth
    mouth_52_truth = groundturth[:, :, 51] - mouth_mean_truth
    mouth_53_truth = groundturth[:, :, 52] - mouth_mean_truth
    mouth_54_truth = groundturth[:, :, 53] - mouth_mean_truth
    mouth_55_truth = groundturth[:, :, 54] - mouth_mean_truth
    mouth_56_truth = groundturth[:, :, 55] - mouth_mean_truth
    mouth_57_truth = groundturth[:, :, 56] - mouth_mean_truth
    mouth_58_truth = groundturth[:, :, 57] - mouth_mean_truth
    mouth_59_truth = groundturth[:, :, 58] - mouth_mean_truth
    mouth_60_truth = groundturth[:, :, 59] - mouth_mean_truth
    mouth_61_truth = groundturth[:, :, 60] - mouth_mean_truth
    mouth_62_truth = groundturth[:, :, 61] - mouth_mean_truth
    # mouth_63_truth = groundturth[:,:,62] - mouth_mean_truth
    # mouth_64_truth = groundturth[:,:,63] - mouth_mean_truth
    # mouth_65_truth = groundturth[:,:,64] - mouth_mean_truth
    # mouth_66_truth = groundturth[:,:,65] - mouth_mean_truth
    # mouth_67_truth = groundturth[:,:,66] - mouth_mean_truth
    # mouth_68_truth = groundturth[:,:,67] - mouth_mean_truth

    # loss+= torch.mean((mouth_48-mouth_48_truth)**2)
    loss += torch.mean((mouth_49 - mouth_49_truth) ** 2)
    loss += torch.mean((mouth_50 - mouth_50_truth) ** 2)
    loss += torch.mean((mouth_51 - mouth_51_truth) ** 2)
    loss += torch.mean((mouth_52 - mouth_52_truth) ** 2)
    loss += torch.mean((mouth_53 - mouth_53_truth) ** 2)
    loss += torch.mean((mouth_54 - mouth_54_truth) ** 2)
    loss += torch.mean((mouth_55 - mouth_55_truth) ** 2)
    loss += torch.mean((mouth_56 - mouth_56_truth) ** 2)
    loss += torch.mean((mouth_57 - mouth_57_truth) ** 2)
    loss += torch.mean((mouth_58 - mouth_58_truth) ** 2)
    loss += torch.mean((mouth_59 - mouth_59_truth) ** 2)
    loss += torch.mean((mouth_60 - mouth_60_truth) ** 2)
    loss += torch.mean((mouth_61 - mouth_61_truth) ** 2)
    loss += torch.mean((mouth_62 - mouth_62_truth) ** 2)
    # loss+= torch.mean((mouth_63-mouth_63_truth)**2)
    # loss+= torch.mean((mouth_64-mouth_64_truth)**2)
    # loss+= torch.mean((mouth_65-mouth_65_truth)**2)
    # loss+= torch.mean((mouth_66-mouth_66_truth)**2)
    # loss+= torch.mean((mouth_67-mouth_67_truth)**2)
    # loss+= torch.mean((mouth_68-mouth_68_truth)**2)

    # loss_angle += cos_Loss(mouth_49, mouth_49_truth, N)
    # loss_angle += cos_Loss(mouth_50, mouth_50_truth, N)
    # loss_angle += cos_Loss(mouth_51, mouth_51_truth, N)
    # loss_angle += cos_Loss(mouth_52, mouth_52_truth, N)
    # loss_angle += cos_Loss(mouth_53, mouth_53_truth, N)
    # loss_angle += cos_Loss(mouth_54, mouth_54_truth, N)
    # loss_angle += cos_Loss(mouth_55, mouth_55_truth, N)
    # loss_angle += cos_Loss(mouth_56, mouth_56_truth, N)
    # loss_angle += cos_Loss(mouth_57, mouth_57_truth, N)
    # loss_angle += cos_Loss(mouth_58, mouth_58_truth, N)
    # loss_angle += cos_Loss(mouth_59, mouth_59_truth, N)
    # loss_angle += cos_Loss(mouth_60, mouth_60_truth, N)
    # loss_angle += cos_Loss(mouth_61, mouth_61_truth, N)
    # loss_angle += cos_Loss(mouth_62, mouth_62_truth, N)
    # loss_angle += cos_Loss(mouth_63, mouth_63_truth, N)
    # loss_angle += cos_Loss(mouth_64, mouth_64_truth, N)
    # loss_angle += cos_Loss(mouth_65, mouth_65_truth, N)
    # loss_angle += cos_Loss(mouth_66, mouth_66_truth, N)
    # loss_angle += cos_Loss(mouth_67, mouth_67_truth, N)
    # loss_angle += cos_Loss(mouth_68, mouth_68_truth, N)

    # ??????
    chin_1_noise = predict[:, :, 0] - predict[:, :, 30]
    chin_2_noise = predict[:, :, 1] - predict[:, :, 30]
    chin_3_noise = predict[:, :, 2] - predict[:, :, 30]
    chin_4_noise = predict[:, :, 3] - predict[:, :, 30]
    chin_5_noise = predict[:, :, 4] - predict[:, :, 30]
    chin_6_noise = predict[:, :, 5] - predict[:, :, 30]
    chin_7_noise = predict[:, :, 6] - predict[:, :, 30]
    chin_8_noise = predict[:, :, 7] - predict[:, :, 30]
    chin_9_noise = predict[:, :, 8] - predict[:, :, 30]
    chin_10_noise = predict[:, :, 9] - predict[:, :, 30]
    chin_11_noise = predict[:, :, 10] - predict[:, :, 30]
    chin_12_noise = predict[:, :, 11] - predict[:, :, 30]
    chin_13_noise = predict[:, :, 12] - predict[:, :, 30]
    chin_14_noise = predict[:, :, 13] - predict[:, :, 30]
    chin_15_noise = predict[:, :, 14] - predict[:, :, 30]
    chin_16_noise = predict[:, :, 15] - predict[:, :, 30]
    chin_17_noise = predict[:, :, 16] - predict[:, :, 30]

    chin_1_noise_truth = groundturth[:, :, 0] - groundturth[:, :, 30]
    chin_2_noise_truth = groundturth[:, :, 1] - groundturth[:, :, 30]
    chin_3_noise_truth = groundturth[:, :, 2] - groundturth[:, :, 30]
    chin_4_noise_truth = groundturth[:, :, 3] - groundturth[:, :, 30]
    chin_5_noise_truth = groundturth[:, :, 4] - groundturth[:, :, 30]
    chin_6_noise_truth = groundturth[:, :, 5] - groundturth[:, :, 30]
    chin_7_noise_truth = groundturth[:, :, 6] - groundturth[:, :, 30]
    chin_8_noise_truth = groundturth[:, :, 7] - groundturth[:, :, 30]
    chin_9_noise_truth = groundturth[:, :, 8] - groundturth[:, :, 30]
    chin_10_noise_truth = groundturth[:, :, 9] - groundturth[:, :, 30]
    chin_11_noise_truth = groundturth[:, :, 10] - groundturth[:, :, 30]
    chin_12_noise_truth = groundturth[:, :, 11] - groundturth[:, :, 30]
    chin_13_noise_truth = groundturth[:, :, 12] - groundturth[:, :, 30]
    chin_14_noise_truth = groundturth[:, :, 13] - groundturth[:, :, 30]
    chin_15_noise_truth = groundturth[:, :, 14] - groundturth[:, :, 30]
    chin_16_noise_truth = groundturth[:, :, 15] - groundturth[:, :, 30]
    chin_17_noise_truth = groundturth[:, :, 16] - groundturth[:, :, 30]

    loss += torch.mean((chin_1_noise - chin_1_noise_truth) ** 2)
    loss += torch.mean((chin_2_noise - chin_2_noise_truth) ** 2)
    loss += torch.mean((chin_3_noise - chin_3_noise_truth) ** 2)
    loss += torch.mean((chin_4_noise - chin_4_noise_truth) ** 2)
    loss += torch.mean((chin_5_noise - chin_5_noise_truth) ** 2)
    loss += torch.mean((chin_6_noise - chin_6_noise_truth) ** 2)
    loss += torch.mean((chin_7_noise - chin_7_noise_truth) ** 2)
    loss += torch.mean((chin_8_noise - chin_8_noise_truth) ** 2)
    loss += torch.mean((chin_9_noise - chin_9_noise_truth) ** 2)
    loss += torch.mean((chin_10_noise - chin_10_noise_truth) ** 2)
    loss += torch.mean((chin_11_noise - chin_11_noise_truth) ** 2)
    loss += torch.mean((chin_12_noise - chin_12_noise_truth) ** 2)
    loss += torch.mean((chin_13_noise - chin_13_noise_truth) ** 2)
    loss += torch.mean((chin_14_noise - chin_14_noise_truth) ** 2)
    loss += torch.mean((chin_15_noise - chin_15_noise_truth) ** 2)
    loss += torch.mean((chin_16_noise - chin_16_noise_truth) ** 2)
    loss += torch.mean((chin_17_noise - chin_17_noise_truth) ** 2)

    # loss_angle += cos_Loss(chin_1_noise, chin_1_noise_truth, N)
    # loss_angle += cos_Loss(chin_2_noise, chin_2_noise_truth, N)
    # loss_angle += cos_Loss(chin_3_noise, chin_3_noise_truth, N)
    # loss_angle += cos_Loss(chin_4_noise, chin_4_noise_truth, N)
    # loss_angle += cos_Loss(chin_5_noise, chin_5_noise_truth, N)
    # loss_angle += cos_Loss(chin_6_noise, chin_6_noise_truth, N)
    # loss_angle += cos_Loss(chin_7_noise, chin_7_noise_truth, N)
    # loss_angle += cos_Loss(chin_8_noise, chin_8_noise_truth, N)
    # loss_angle += cos_Loss(chin_9_noise, chin_9_noise_truth, N)
    # loss_angle += cos_Loss(chin_10_noise, chin_10_noise_truth, N)
    # loss_angle += cos_Loss(chin_11_noise, chin_11_noise_truth, N)
    # loss_angle += cos_Loss(chin_12_noise, chin_12_noise_truth, N)
    # loss_angle += cos_Loss(chin_13_noise, chin_13_noise_truth, N)
    # loss_angle += cos_Loss(chin_14_noise, chin_14_noise_truth, N)
    # loss_angle += cos_Loss(chin_15_noise, chin_15_noise_truth, N)
    # loss_angle += cos_Loss(chin_16_noise, chin_16_noise_truth, N)
    # loss_angle += cos_Loss(chin_17_noise, chin_17_noise_truth, N)
    # loss_angle1 += angle_cos_Loss(chin_1_noise,chin_17_noise,chin_1_noise_truth,chin_17_noise_truth,N)
    # loss_angle1 += angle_cos_Loss(chin_2_noise,chin_16_noise,chin_2_noise_truth,chin_16_noise_truth,N)
    # loss_angle1 += angle_cos_Loss(chin_3_noise,chin_15_noise,chin_3_noise_truth,chin_15_noise_truth,N)
    # loss_angle1 += angle_cos_Loss(chin_4_noise,chin_14_noise,chin_4_noise_truth,chin_14_noise_truth,N)
    # loss_angle1 += angle_cos_Loss(chin_5_noise,chin_13_noise,chin_5_noise_truth,chin_13_noise_truth,N)
    # loss_angle1 += angle_cos_Loss(chin_6_noise,chin_12_noise,chin_6_noise_truth,chin_12_noise_truth,N)
    # loss_angle1 += angle_cos_Loss(chin_7_noise,chin_11_noise,chin_7_noise_truth,chin_11_noise_truth,N)
    # loss_angle1 += angle_cos_Loss(chin_8_noise,chin_10_noise,chin_8_noise_truth,chin_10_noise_truth,N)

    chin_1_chin = predict[:, :, 0] - predict[:, :, 8]
    chin_2_chin = predict[:, :, 1] - predict[:, :, 8]
    chin_3_chin = predict[:, :, 2] - predict[:, :, 8]
    chin_4_chin = predict[:, :, 3] - predict[:, :, 8]
    chin_5_chin = predict[:, :, 4] - predict[:, :, 8]
    chin_6_chin = predict[:, :, 5] - predict[:, :, 8]
    chin_7_chin = predict[:, :, 6] - predict[:, :, 8]
    chin_8_chin = predict[:, :, 7] - predict[:, :, 8]
    chin_10_chin = predict[:, :, 9] - predict[:, :, 8]
    chin_11_chin = predict[:, :, 10] - predict[:, :, 8]
    chin_12_chin = predict[:, :, 11] - predict[:, :, 8]
    chin_13_chin = predict[:, :, 12] - predict[:, :, 8]
    chin_14_chin = predict[:, :, 13] - predict[:, :, 8]
    chin_15_chin = predict[:, :, 14] - predict[:, :, 8]
    chin_16_chin = predict[:, :, 15] - predict[:, :, 8]
    chin_17_chin = predict[:, :, 16] - predict[:, :, 8]

    chin_1_chin_truth = groundturth[:, :, 0] - groundturth[:, :, 8]
    chin_2_chin_truth = groundturth[:, :, 1] - groundturth[:, :, 8]
    chin_3_chin_truth = groundturth[:, :, 2] - groundturth[:, :, 8]
    chin_4_chin_truth = groundturth[:, :, 3] - groundturth[:, :, 8]
    chin_5_chin_truth = groundturth[:, :, 4] - groundturth[:, :, 8]
    chin_6_chin_truth = groundturth[:, :, 5] - groundturth[:, :, 8]
    chin_7_chin_truth = groundturth[:, :, 6] - groundturth[:, :, 8]
    chin_8_chin_truth = groundturth[:, :, 7] - groundturth[:, :, 8]
    chin_10_chin_truth = groundturth[:, :, 9] - groundturth[:, :, 8]
    chin_11_chin_truth = groundturth[:, :, 10] - groundturth[:, :, 8]
    chin_12_chin_truth = groundturth[:, :, 11] - groundturth[:, :, 8]
    chin_13_chin_truth = groundturth[:, :, 12] - groundturth[:, :, 8]
    chin_14_chin_truth = groundturth[:, :, 13] - groundturth[:, :, 8]
    chin_15_chin_truth = groundturth[:, :, 14] - groundturth[:, :, 8]
    chin_16_chin_truth = groundturth[:, :, 15] - groundturth[:, :, 8]
    chin_17_chin_truth = groundturth[:, :, 16] - groundturth[:, :, 8]

    # loss+= torch.mean((chin_1_chin-chin_1_chin_truth)**2)
    # loss+= torch.mean((chin_2_chin-chin_2_chin_truth)**2)
    # loss+= torch.mean((chin_3_chin-chin_3_chin_truth)**2)
    # loss+= torch.mean((chin_4_chin-chin_4_chin_truth)**2)
    # loss+= torch.mean((chin_5_chin-chin_5_chin_truth)**2)
    # loss+= torch.mean((chin_6_chin-chin_6_chin_truth)**2)
    # loss+= torch.mean((chin_7_chin-chin_7_chin_truth)**2)
    # loss+= torch.mean((chin_8_chin-chin_8_chin_truth)**2)
    # loss+= torch.mean((chin_17_chin-chin_17_chin_truth)**2)
    # loss+= torch.mean((chin_10_chin-chin_10_chin_truth)**2)
    # loss+= torch.mean((chin_11_chin-chin_11_chin_truth)**2)
    # loss+= torch.mean((chin_12_chin-chin_12_chin_truth)**2)
    # loss+= torch.mean((chin_13_chin-chin_13_chin_truth)**2)
    # loss+= torch.mean((chin_14_chin-chin_14_chin_truth)**2)
    # loss+= torch.mean((chin_15_chin-chin_15_chin_truth)**2)
    # loss+= torch.mean((chin_16_chin-chin_16_chin_truth)**2)

    # loss_angle += cos_Loss(chin_1_chin, chin_1_chin_truth, N)
    # loss_angle += cos_Loss(chin_2_chin, chin_2_chin_truth, N)
    # loss_angle += cos_Loss(chin_3_chin, chin_3_chin_truth, N)
    # loss_angle += cos_Loss(chin_4_chin, chin_4_chin_truth, N)
    # loss_angle += cos_Loss(chin_5_chin, chin_5_chin_truth, N)
    # loss_angle += cos_Loss(chin_6_chin, chin_6_chin_truth, N)
    # loss_angle += cos_Loss(chin_7_chin, chin_7_chin_truth, N)
    # loss_angle += cos_Loss(chin_8_chin, chin_8_chin_truth, N)
    # loss_angle += cos_Loss(chin_15_chin, chin_15_chin_truth, N)
    # loss_angle += cos_Loss(chin_10_chin, chin_10_chin_truth, N)
    # loss_angle += cos_Loss(chin_11_chin, chin_11_chin_truth, N)
    # loss_angle += cos_Loss(chin_12_chin, chin_12_chin_truth, N)
    # loss_angle += cos_Loss(chin_13_chin, chin_13_chin_truth, N)
    # loss_angle += cos_Loss(chin_14_chin, chin_14_chin_truth, N)
    # loss_angle += cos_Loss(chin_16_chin, chin_15_chin_truth, N)

    # chin_1_17 = predict[:,:,0] - predict[:,:,16]
    # chin_2_16 = predict[:,:,1] - predict[:,:,15]
    # chin_3_15 = predict[:,:,2] - predict[:,:,14]
    # chin_4_14 = predict[:,:,3] - predict[:,:,13]
    # chin_5_13 = predict[:,:,4] - predict[:,:,12]
    # chin_6_12 = predict[:,:,5] - predict[:,:,11]
    # chin_7_11 = predict[:,:,6] - predict[:,:,10]
    # chin_8_10 = predict[:,:,7] - predict[:,:,9]

    # chin_1_17_truth = groundturth[:,:,0] - groundturth[:,:,16]
    # chin_2_16_truth = groundturth[:,:,1] - groundturth[:,:,15]
    # chin_3_15_truth = groundturth[:,:,2] - groundturth[:,:,14]
    # chin_4_14_truth = groundturth[:,:,3] - groundturth[:,:,13]
    # chin_5_13_truth = groundturth[:,:,4] - groundturth[:,:,12]
    # chin_6_12_truth = groundturth[:,:,5] - groundturth[:,:,11]
    # chin_7_11_truth = groundturth[:,:,6] - groundturth[:,:,10]
    # chin_8_10_truth = groundturth[:,:,7] - groundturth[:,:,9]

    # loss += torch.mean((chin_1_17 - chin_1_17_truth)**2)
    # loss += torch.mean((chin_2_16 - chin_2_16_truth)**2)
    # loss += torch.mean((chin_3_15 - chin_3_15_truth)**2)
    # loss += torch.mean((chin_4_14 - chin_4_14_truth)**2)
    # loss += torch.mean((chin_5_13 - chin_5_13_truth)**2)
    # loss += torch.mean((chin_6_12 - chin_6_12_truth)**2)
    # loss += torch.mean((chin_7_11 - chin_7_11_truth)**2)
    # loss += torch.mean((chin_8_10 - chin_8_10_truth)**2)

    # print(loss_angle*1000)
    # print(loss)
    # print(loss_angle1*1000)
    face_0 = predict[:, :, 0]
    face_0_t = groundturth[:, :, 0]
    face_17 = predict[:, :, 17]
    face_17_t = groundturth[:, :, 17]
    face_8 = predict[:, :, 8]
    face_8_t = groundturth[:, :, 8]
    n_30 = predict[:, :, 30]
    n_30_t = groundturth[:, :, 30]

    #Ll = ((face_0 + face_8 + left_eye_mean) / 3) - n_30 * 3
    Ll = ((face_0 + face_8 + left_eye_mean) / 3) - n_30
    #Ll_t = ((face_0_t + face_8_t + left_eye_mean_truth) / 3) - n_30_t * 3
    Ll_t = ((face_0_t + face_8_t + left_eye_mean_truth) / 3) - n_30_t
    #Lr = ((face_17 + face_8 + right_eye_mean) / 3) - n_30 * 3
    Lr = ((face_17 + face_8 + right_eye_mean) / 3) - n_30
    #Lr_t = ((face_17_t + face_8_t + right_eye_mean_truth) / 3) - n_30_t * 3
    Lr_t = ((face_17_t + face_8_t + right_eye_mean_truth) / 3) - n_30_t
    a = torch.abs(Ll_t - Ll)
    b = torch.abs(Lr_t - Lr)
    s = torch.abs(a - b)
    loss += torch.mean(s**2)  #**2

    # return loss*0.95+loss_angle*1000
    return loss * 0.00195
