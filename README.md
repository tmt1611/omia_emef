# EMEF: Ensemble Multi-Exposure Image Fusion, AAAI 2023 <a href="https://arxiv.org/abs/2305.12734"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> 

## Pretrained Model

[Google Drive](https://drive.google.com/drive/folders/13fIAHG2yAgCIoegbA3mv2jP1ZSglG-nR?usp=drive_link) | [Baidu Netdisk](https://pan.baidu.com/s/1m1ijn6o93mIJ_hsoAxlanw?pwd=emef) (code: emef)

## Setting up : Clone the repo, install requirements, download weights
Here is an example
```
# Clone
!git clone https://github.com/tmt1611/omia_emef
%cd omia_emef
# Install dependencies
!pip install torch torchvision torchaudio
!pip install -r requirements.txt
# pretrained
!mkdir -p inference
!wget --header="Host: drive.usercontent.google.com" \
--header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36" \
--header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" \
--header="Accept-Language: en-US,en;q=0.9,vi;q=0.8,fr;q=0.7" \
--header="Cookie: HSID=AQhDlpWc7SNSu_jh_; SSID=AI79y8qRwSfA64tYE; APISID=ODligCOOj2R66HTZ/A1Rc8T6NUvrD7SzAF; SAPISID=ADUSIWKhThEdPNaH/AnPJBwVpFR6_pF1l3; __Secure-1PAPISID=ADUSIWKhThEdPNaH/AnPJBwVpFR6_pF1l3; __Secure-3PAPISID=ADUSIWKhThEdPNaH/AnPJBwVpFR6_pF1l3; OGPC=19015969-1:19010599-1:; OGP=-19015969:-19010599:; __Secure-BUCKET=CI4E; SID=g.a000kQgKyCIGbx_qufNoSEU9DXrRSIuwm7DxRnbp4vL-stXDIITyhFYv1o0SvLF2nKkE6eXK5wACgYKAaUSARUSFQHGX2MifN8L9qNE2iU-m6H-hhrFHBoVAUF8yKoJ-32dfF-t8cinuKg8K2Wg0076; __Secure-1PSID=g.a000kQgKyCIGbx_qufNoSEU9DXrRSIuwm7DxRnbp4vL-stXDIITyU4KZ4bmtx3MX3sM64S2VnwACgYKAYYSARUSFQHGX2MioLAFB_ZvR7t-BZ0lbK03rBoVAUF8yKr3fW5g5Fd9u-nfEZV3ZxTM0076; __Secure-3PSID=g.a000kQgKyCIGbx_qufNoSEU9DXrRSIuwm7DxRnbp4vL-stXDIITyRZ0PYlF4Bl_4a220zHs-BgACgYKAZQSARUSFQHGX2Miqi8_tlqGOsNGl2KBkKee-hoVAUF8yKplinagpI86LUyIvYCKxHm10076; AEC=AQTF6Hx3qwK-VAMFCEjrR_EaVpqyURD0Q-WNSezOjyU_aRTJ9VK_i9rkYw; SEARCH_SAMESITE=CgQIp5sB; NID=514=OUZ7gBUd_8rtrl2jXhHMe1lGPVGyho21ukWI_V668cjM9q_YfiTGdN_Z9ISae0GL1vLFn31cWtofJ-ImrDd_3LYEzb5NNPW5k7T01MReYAY9erFNFV_2xIdH1Cnko7qlicdzpW-JpJ-ua1EY0taQXBVrwoxBv9VKiEnrx3hRiXCA2VJAAOC3Ouqbo-TlZtOteH-HIt4dW75r7GjnrvHRmX_HwL2V-W19fuTAHNdJs4umSCj_JqkesPRaArO7JLZfLWhM2u84zKM6SFbK9uH-tPZ1Z5dead6kyGcQMyq2YOdPD86VIyhh1_CzUMayF_3WavuI52p9iXk7kBV_6e2IaH7YVrXTt62APg3VSJOAGjBgLHU63Gj3wneC9sCZs1av0akJGJZJaC-c6pfZlUD3pf_zbxLn64b8U-IHgMrxNmtF5RkHkcE; __Secure-ENID=20.SE=EmWOYq-A_5N9bXgiXxJQ6GWeYsaL0NpCJDQGJLyGXjDgtAIppgAVPQPeGJvkr0yqlru9aqhckrhH3LWefyRbj8C2vgU90HVps56b20pr3ilk2NABmY8qCFRp2wl86W5qmDJVI3aD4ZN4m7yvyv3uRibovbjq-KhGV0_LNlSjikz6rG6_Okt0rdr41dvIjQ2BbhZBpHRuVTy0kRt36RNvu5jgvDwTpzET2nD0nGwNdkeej5W9-933D3Zhj_uR9DV6-_1PprVIf38m8pDInxz-uYGgoSLRCu6GmxYX4q5UUZhxpCEYy5_954o7JfFOrx2qU46W8ZtjWX08zLTVy-nBnMg-V6oTGaZymyvfBOPhix6u; __Secure-1PSIDTS=sidts-CjEB3EgAEgCVxTZUDnMDCWQPkFfDypj7UGYuBKXHXy_pLoYuV_O9yspovh8mdLg6twvZEAA; __Secure-3PSIDTS=sidts-CjEB3EgAEgCVxTZUDnMDCWQPkFfDypj7UGYuBKXHXy_pLoYuV_O9yspovh8mdLg6twvZEAA; SIDCC=AKEyXzU0zL9cvnXjuwuLrqEywmG3JqFxmNjcnmNz6ZEUmzYWKW-YNX5YiSDGHGHOmh9Pfm1f02w; __Secure-1PSIDCC=AKEyXzV_Np3gDDLC806LXlvo7pwqRXbg_XTm3ZWHJRAINtZfWu2096dIGDPp87rrbGHWdZSc9Lnr; __Secure-3PSIDCC=AKEyXzVYpgxecl5llvWbGFPcr5nffjTKvy-1Ftu-WyO4ym6S0v_mBtMUj_MOSiq8ycPqipJzhJc" --header="Connection: keep-alive" "https://drive.usercontent.google.com/download?id=1RY3qARQXVJK2CyQ72GHECDrLy5R6rpW4&export=download&authuser=0&confirm=t&uuid=8b889bca-a06b-41ce-9724-301a4f10d288&at=APZUnTVdLmd4yyCYhQkCl6u3LeL1:1717584407046" \
-c -O 'inference/latest_net_G.pth' 
```

## Datasets
Structured as below. The data folder dataroot can be precised when running inference (section Inference).
```
- dataroot
  - oe
    - 1.png
    - 2.png
    ...
  - ue
    - 1.png
    - 2.png
    ...
```
## Inference
```
!python inference.py \
    --dataroot test_data \
    --name inference \
    --model infer \
    --phase test \
    --dataset_mode HDR \
    --checkpoints_dir . \
    --results_dir ./results \
    # --gray 0 \
    --epoch latest
```
dataroot : folder containing data

name : name of the experiment, should has the same name as the folder containing the pretrained weights.

model : name of script containing the model (class) used

results_dir : folder containing the output

epoch : epoch of the pretrained model to be loaded

## Options
Further options can be found in the classes contained in options/base_options.py, options/test_options.py, models/infer_models.py, models/base_models.py

## Inference code
I wrote the infer_model.py and inference.py based on the paper and the models/base_models.py, models/demo_models.py. Also modified data/HDR_dataset.py to generate 'cls' to feed to model.netG in infer_model.py.

