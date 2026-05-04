from typing import Dict, Union
import numpy.typing as npt
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import colorednoise as cn
import os

Numeric = Union[int, float]

class AddShear(object):
    # Add shear jumps in signal
    def __call__(self, sample: Dict,seed=None):
        noisy, clean, weight = sample['noisy'], sample['clean'], sample["weight"]
        if seed is None:
          rng = np.random.default_rng()
        else:
          rng = np.random.default_rng(seed)

        max_size = np.max(noisy)-np.min(noisy)
        shear_pts = np.array(rng.integers(1, len(noisy), size = [rng.integers(1, 3, dtype = int),]))
        sheared_noisy = noisy.copy()
        for shear_pt in shear_pts:
            shear_size = (rng.random()-0.5)*2*max_size

            shear_sig = np.concatenate([np.zeros(shear_pt), shear_size * np.ones(len(noisy) - shear_pt)])
            sheared_noisy = sheared_noisy+shear_sig

        return {'noisy': sheared_noisy,
                'clean': clean,
                'weight': weight}


class AddColorNoise(object):
    def __call__(self, sample: Dict, beta: npt.ArrayLike, snr: Numeric,seed=None):
        noisy, clean, weight = sample['noisy'], sample['clean'], sample["weight"]
        if seed is None:
          color_noise = cn.powerlaw_psd_gaussian(beta, noisy.size)
        else:
          color_noise = cn.powerlaw_psd_gaussian(beta, noisy.size, random_state=seed)
        cur_snr = 10*np.log10(np.sum(np.square(clean))/np.sum(np.square(color_noise)))
        scale = np.sqrt(10**(-1*(snr-cur_snr)/10))
        return {'noisy': noisy+(scale*color_noise),
                'clean': clean,
                'weight': weight}

class AddImpulse(object):
    def __call__(self, sample: Dict, max_amplitude=5.0, max_width = 5,seed=None):
        noisy = sample['noisy']
        if seed is None:
          rng = np.random.default_rng()
        else:
          rng = np.random.default_rng(seed)

        num_spikes = rng.integers(1, 4)
        spike_pts = rng.integers(0, len(noisy) - max_width, size=num_spikes)

        noisy_spiked = noisy.copy()
        signal_std = np.std(sample['clean']) + 1e-6

        for pt in spike_pts:
            direction = rng.choice([-1, 1])
            strength = max_amplitude * signal_std
            width = rng.integers(1, max_width + 1)
            envelope = np.hanning(width)

            noisy_spiked[pt : pt + width] += (direction * strength * envelope)

        return {'noisy': noisy_spiked, 'clean': sample['clean'], 'weight': sample['weight']}



class AddClipping(object):
    # Simulates Saturation (Signal hits min/max voltage rails)
    def __call__(self, sample: Dict, max_percentile = 80,seed=None):
        if seed is None:
          rng = np.random.default_rng()
        else:
          rng = np.random.default_rng(seed)
        noisy = sample['noisy']

        high_rail = np.percentile(noisy, rng.uniform(max_percentile, 95))
        clipped_signal = np.minimum(noisy, high_rail)

        return {'noisy': clipped_signal, 'clean': sample['clean'], 'weight': sample['weight']}


shear_augmentor = AddShear()
color_augmentor = AddColorNoise()
impulse_augmentor = AddImpulse()
clipping_augmentor = AddClipping()

def noise_augmentation(clean, noise_bank, num_epochs = 10, max_epoch = 100, for_valid = False, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    N, L_clean = clean.shape

    noisy = np.zeros_like(clean)
    target = np.zeros_like(clean)
    progress = num_epochs / max_epoch

    use_addictive = use_noisebank = use_color = use_shear = use_impulse = use_gain = use_offset = use_discontinued = use_clipping = False
    selected_problems = []

    if for_valid == True:
      selected_problems.append(rng.choice(['color','additive_noise']))
      selected_problems.append(rng.choice(['shear','clipping','impulse']))
    else:
      if rng.random() < 0.8:
        selected_problems.append(rng.choice(['color','additive_noise']))
        selected_problems.append(rng.choice(['shear','clipping','impulse']))


    use_gain         = 'gain' in selected_problems
    use_addictive    = 'additive_noise' in selected_problems
    use_noisebank    = 'noisebank' in selected_problems
    use_color        = 'color' in selected_problems
    use_shear        = 'shear' in selected_problems
    use_impulse      = 'impulse' in selected_problems
    use_discontinued = 'discontinued' in selected_problems
    use_clipping     = 'clipping' in selected_problems
    use_saturation     = 'saturation' in selected_problems

    for i in range(N):
        current_noise_signal = clean[i]

        # 1. addictive noise
        if use_addictive:
          noise = noise_bank['addictive_noise']
          N_noise = noise.shape[0]
          max_offset = noise.shape[1]

          if for_valid == True:
            noise_idx = rng.integers(int(N_noise * 0.8), N_noise)
            amplitude = np.std(clean[i]) + 1e-6
            amplitude = amplitude * rng.uniform(0.7, 1.3)
            shift = rng.integers(0, max_offset)
            
          else:
            noise_idx = rng.integers(0, int(N_noise * 0.8))
            amplitude = np.std(clean[i]) + 1e-6
            amplitude = amplitude * rng.uniform(0.7, 1.3)
            shift = rng.integers(0, max_offset)

          # noise_idx = rng.integers(0, N_noise)
          noise = np.roll(noise[noise_idx, :].copy(), shift)

          current_noise_signal = clean[i].copy() + (noise * amplitude)

        # 2. Colored noise
        if use_color:
            sample_dict = {
                'noisy': current_noise_signal,
                'clean': clean[i],
                'weight': 1
            }
            if for_valid == True:
                color_beta = rng.uniform(0, 2)
                color_snr_db = rng.uniform(20,30)
                seed = rng.integers(0, 1000000)
            else:
                color_beta = rng.uniform(0, 2)
                color_snr_db = rng.uniform(20,30)
                seed = None

            result_dict = color_augmentor(sample_dict, beta=color_beta, snr=color_snr_db, seed = seed)

            current_noise_signal = result_dict['noisy']

        # 3. Shear
        if use_shear:
            if for_valid == True:
              seed = rng.integers(0, 1000000)
            else:
              seed = None

            sample_dict = {'noisy': current_noise_signal, 'clean': clean[i], 'weight': 1}
            result = shear_augmentor(sample_dict, seed = seed)
            current_noise_signal = result['noisy']

        # 4. Impulse
        if use_impulse:
            if for_valid == True:
              impulse_amplitude = rng.uniform(5,30)
              impulse_width = 11
              seed = rng.integers(0, 1000000)
            else:
              impulse_amplitude = rng.uniform(5,30)
              impulse_width = 11
              seed = None

            sample_dict = {'noisy': current_noise_signal, 'clean': clean[i], 'weight': 1}
            result = impulse_augmentor(sample_dict,max_amplitude = impulse_amplitude, max_width=impulse_width, seed = seed)
            current_noise_signal = result['noisy']

        # 5. clipping
        if use_clipping:
            if for_valid == True:
              seed = rng.integers(0, 1000000)
            else:
              seed = None

            sample_dict = {'noisy': current_noise_signal, 'clean': clean[i], 'weight': 1}
            result = clipping_augmentor(sample_dict,max_percentile=80,seed = seed)
            current_noise_signal = result['noisy']


        noisy[i] = current_noise_signal
        target[i] = clean[i]

    return target, noisy, selected_problems

def subject_split(target, subject, noisy=None, test_ratio=0.2, seed=99):
    # Ensure subject is 1D
    subject = np.squeeze(subject)
    unique_subj = np.unique(subject)

    # Split unique subjects
    train_subj, test_subj = train_test_split(
        unique_subj, test_size=test_ratio, random_state=seed
    )

    # Create masks based on subject ID
    train_mask = np.isin(subject, train_subj)
    test_mask  = np.isin(subject, test_subj)

    # Split required arrays
    target_train, target_test = target[train_mask], target[test_mask]
    subj_train, subj_test   = subject[train_mask], subject[test_mask]

    # Conditional return
    if noisy is not None:
        noisy_train, noisy_test = noisy[train_mask], noisy[test_mask]
        return (
            target_train, target_test,
            noisy_train, noisy_test,
            subj_train, subj_test
        )
    else:
        return (
            target_train, target_test,
            subj_train, subj_test
        )
    
def print_dataset_info(name, target, subject=None):
    print(f"\n{name}")
    print("-" * 40)
    print(f"Segments : {target.shape[0]}")
    print(f"Length   : {target.shape[1]}")
    
    if subject is not None:
        subject = np.squeeze(subject)
        unique_subj = np.unique(subject)
        print(f"Subjects : {len(unique_subj)}")
    else:
        print("Subjects : Not available")
        
def subject_split_and_stack(dataset_path, test_ratio=0.1, seed=99, load_all=False):
    data_path1 = os.path.join(dataset_path, 'processed_datasets.pt')
    print("📥 Loading dataset:", data_path1)
    data = torch.load(data_path1, weights_only=False)

    # ----------------------
    # Public dataset
    # ----------------------
    noise_bank = {
        'addictive_noise':  data['addictive_noise'],
        'noise_filt_cmad1': data['noise_filt_cmad1'],
        'noise_filt_cmad2': data['noise_filt_cmad2'],
        'noise_filt_umad1': data['noise_filt_umad1'],
        'noise_filt_umad2': data['noise_filt_umad2'],
        'noise_imu_cmad1':  data['noise_imu_cmad1'],
        'noise_imu_umad1':  data['noise_imu_umad1']
    }

    target_public  = data['target_public']
    subject_public = data['subject_public']

    # ----------------------
    # CMAD1 dataset
    # ----------------------
    noisy_CMAD1     = data['noisy_CMAD1']
    noisy_GAN_CMAD1 = data['noisy_GAN_CMAD1']
    target_CMAD1    = data['target_CMAD1']
    subject_CMAD1   = data['subject_CMAD1']

    # ----------------------
    # CMAD2 dataset
    # ----------------------
    noisy_CMAD2     = data['noisy_CMAD2']
    noisy_GAN_CMAD2 = data['noisy_GAN_CMAD2']
    target_CMAD2    = data['target_CMAD2']
    subject_CMAD2   = data['subject_CMAD2']

    # ----------------------
    # UMAC1 dataset
    # ----------------------
    noisy_UMAD1    = data['noisy_UMAD1']
    noisy_GAN_UMAD1= data['noisy_GAN_UMAD1']
    target_UMAD1   = data['target_UMAD1']
    subject_UMAD1  = data['subject_UMAD1']

    # ----------------------
    # UMAC2 dataset
    # ----------------------
    noisy_UMAD2    = data['noisy_UMAD2']
    noisy_GAN_UMAD2= data['noisy_GAN_UMAD2']
    target_UMAD2   = data['target_UMAD2']
    subject_UMAD2  = data['subject_UMAD2']

    # ----------------------
    # CNSOT1 & CNSOT2 dataset
    # ----------------------
    noisy_CNSOT1   = data['noisy_CNSOT1']
    subject_CNSOT1 = data['subject_CNSOT1']
    label_CNSOT1   = data['label_CNSOT1']

    noisy_CNSOT2   = data['noisy_CNSOT2']
    subject_CNSOT2 = data['subject_CNSOT2']
    label_CNSOT2   = data['label_CNSOT2']

    print("✔ All datasets loaded successfully!")

    # Data save
    if(load_all == False):
        print("Splitting CMAD dataset...")
        CMAD_target_tr, CMAD_target_te, CMAD_subj_tr, CMAD_subj_te = \
            subject_split(target_CMAD1, subject_CMAD1, test_ratio=test_ratio, seed=seed)

        print("Splitting UMAD1 dataset...")
        UMAD1_target_tr, UMAD1_target_te, UMAD1_subj_tr, UMAD1_subj_te = \
            subject_split(target_UMAD1, subject_UMAD1, test_ratio=test_ratio, seed=seed)

        print("Splitting UMAD2 dataset...")
        UMAD2_target_tr, UMAD2_target_te, UMAD2_subj_tr, UMAD2_subj_te = \
            subject_split(target_UMAD2, subject_UMAD2, test_ratio=test_ratio, seed=seed)

        print("Splitting PUBLIC dataset...")
        pub_target_tr, pub_target_te, pub_subj_tr, pub_subj_te = \
            subject_split(target_public, subject_public, test_ratio=test_ratio, seed=seed)

        val_rng = np.random.default_rng(seed)
        N, L = np.shape(pub_target_te)
        pub_target_te_final = np.zeros((N, L))
        pub_noisy_te_final = np.zeros((N, L))
        problem_log = []
        for i in range(N):
            target_sample, noisy_sample, current_problems = noise_augmentation(
                pub_target_te[i:i+1],
                noise_bank,
                for_valid=True,
                rng=val_rng
            )
            # Store the result back
            pub_target_te_final[i, :] = target_sample.squeeze()
            pub_noisy_te_final[i, :] = noisy_sample.squeeze()
            problem_log.append(current_problems)


        N, L = np.shape(CMAD_target_te)
        CMAD_target_te_final = np.zeros((N, L))
        CMAD_noisy_te_final = np.zeros((N, L))
        for i in range(N):
            target_sample, noisy_sample, current_problems = noise_augmentation(
                CMAD_target_te[i:i+1],
                noise_bank,
                for_valid=True,
                rng=val_rng
            )
            # Store the result back
            CMAD_target_te_final[i, :] = target_sample.squeeze()
            CMAD_noisy_te_final[i, :] = noisy_sample.squeeze()
            problem_log.append(current_problems)

        # N, L = np.shape(UMAD1_target_te)
        # UMAD1_target_te_final = np.zeros((N, L))
        # UMAD1_noisy_te_final = np.zeros((N, L))
        # for i in range(N):
        #     target_sample, noisy_sample, current_problems = noise_augmentation(
        #         UMAD1_target_te[i:i+1],
        #         noise_bank,
        #         for_valid=True,
        #         rng=val_rng
        #     )
        #     # Store the result back
        #     UMAD1_target_te_final[i, :] = target_sample.squeeze()
        #     UMAD1_noisy_te_final[i, :] = noisy_sample.squeeze()
        #     problem_log.append(current_problems)

        N, L = np.shape(UMAD2_target_te)
        UMAD2_target_te_final = np.zeros((N, L))
        UMAD2_noisy_te_final = np.zeros((N, L))
        for i in range(N):
            target_sample, noisy_sample, current_problems = noise_augmentation(
                UMAD2_target_te[i:i+1],
                noise_bank,
                for_valid=True,
                rng=val_rng
            )
            # Store the result back
            UMAD2_target_te_final[i, :] = target_sample.squeeze()
            UMAD2_noisy_te_final[i, :] = noisy_sample.squeeze()
            problem_log.append(current_problems)

        cache_file = 'validation_set.pt'
        print(f"Saving validation set to {cache_file}...")
        torch.save({
                        'pub_target_tr': pub_target_tr,
                        'pub_target_te_final': pub_target_te_final,
                        'pub_noisy_te_final': pub_noisy_te_final,
                        'CMAD_target_tr': CMAD_target_tr,
                        'CMAD_target_te_final': CMAD_target_te_final,
                        'CMAD_noisy_te_final': CMAD_noisy_te_final,
                        # 'UMAD1_target_tr': UMAD1_target_tr,
                        # 'UMAD1_target_te_final': UMAD1_target_te_final,
                        # 'UMAD1_noisy_te_final': UMAD1_noisy_te_final,
                        'UMAD2_target_tr': UMAD2_target_tr,
                        'UMAD2_target_te_final': UMAD2_target_te_final,
                        'UMAD2_noisy_te_final': UMAD2_noisy_te_final,
                        'problem_log': problem_log
                        }, cache_file)

    # Data load
    else:
        data_path2 = os.path.join(dataset_path, 'validation_set.pt')
        print("📥 Loading dataset:", data_path2)
        data = torch.load(data_path2, weights_only=False)
        pub_target_tr = data['pub_target_tr']
        pub_noisy_te_final = data['pub_noisy_te_final']
        pub_target_te_final = data['pub_target_te_final']
        CMAD_target_tr = data['CMAD_target_tr']
        CMAD_noisy_te_final = data['CMAD_noisy_te_final']
        CMAD_target_te_final = data['CMAD_target_te_final']
        # UMAD1_target_tr = data['UMAD1_target_tr']
        # UMAD1_target_te_final = data['UMAD1_target_te_final']
        # UMAD1_noisy_te_final = data['UMAD1_noisy_te_final']
        UMAD2_target_tr = data['UMAD2_target_tr']
        UMAD2_target_te_final = data['UMAD2_target_te_final']
        UMAD2_noisy_te_final = data['UMAD2_noisy_te_final']
        problem_log = data['problem_log']
        
        print_dataset_info(
            "PUBLIC - TRAIN",
            pub_target_tr,
            data.get('pub_subj_tr', None)
        )

        print_dataset_info(
            "PUBLIC - VALID",
            pub_target_te_final,
            data.get('pub_subj_te', None)
        )

        print_dataset_info(
            "CMAD - TRAIN",
            CMAD_target_tr,
            data.get('CMAD_subj_tr', None)
        )

        print_dataset_info(
            "CMAD - VALID",
            CMAD_target_te_final,
            data.get('CMAD_subj_te', None)
        )

        print_dataset_info(
            "UMAD2 - TRAIN",
            UMAD2_target_tr,
            data.get('UMAD2_subj_tr', None)
        )

        print_dataset_info(
            "UMAD2 - VALID",
            UMAD2_target_te_final,
            data.get('UMAD2_subj_te', None)
        )

    noisy_train = np.vstack([
        pub_target_tr,
        target_CMAD2,
        CMAD_target_tr,
        UMAD2_target_tr,
    ])

    target_train = np.vstack([
        pub_target_tr,
        target_CMAD2,
        CMAD_target_tr,
        UMAD2_target_tr,
    ])

    noisy_test = np.vstack([
        pub_noisy_te_final,
        CMAD_noisy_te_final,
        UMAD2_noisy_te_final,
    ])

    target_test = np.vstack([
        pub_target_te_final,
        CMAD_target_te_final,
        UMAD2_target_te_final,
    ])

    independent_test = np.vstack([
        noisy_GAN_UMAD1,
        noisy_GAN_UMAD2,
        noisy_CNSOT1,
        noisy_CNSOT2
    ])
    # ---------------------
    # Summary
    # ---------------------
    print("\nDATASET SPLIT SUMMARY")
    print("----------------------------------")
    print(f"Train Noisy : {noisy_train.shape}")
    print(f"Train Clean : {target_train.shape}")
    print("----------------------------------")
    print(f"Test Noisy  : {noisy_test.shape}")
    print(f"Test Clean  : {target_test.shape}")
    print("----------------------------------")
    print(f"Independent Test (Noisy): {independent_test.shape}")
    print("----------------------------------")

    return (
        noisy_train, noisy_test,
        target_train, target_test,
        independent_test, noise_bank, problem_log
    )

