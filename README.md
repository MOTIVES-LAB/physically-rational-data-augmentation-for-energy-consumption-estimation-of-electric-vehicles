# Physically rational data augmentation and validation
Code for the paper **Physically rational data augmentation for energy consumption estimation of electric vehicles**.
## Overview
The physically rational data augmentation and validation is a Python-based program designed to realize the data augmentation of the vehicle driving data in real world. This tool through sample split and sample concatenation of any driving data achieved data augmentation and verify the effectiveness of the proposed method in existing energy consumption estimation models.

## Features
- **Sample Feature Extraction**: Extract sample features from raw trip data.
- **Proposed method for data augmentation**: The proposed data augmentation method is used to amplify the original sample data.
- **Baseline method for data augmentation**: Two existing data augmentation methods to amplify raw sample data.
- **Random Search Parameter Adjustment**: Stochastic search is utilized to achieve the tuning of the hyperparameters of each energy estimation model under the same parameter space.
- **Five fold cross validation**: Existing energy estimation models are trained using augmented data to validate the effectiveness of the proposed augmentation method under five-fold cross-validation conditions.

## Usage

```
###############################################################################
### Tesla model 3 rolling resistance parameters
###############################################################################


para_a=179.40
para_b=0.2800
para_c=0.02350

###############################################################################
### Other parameter settings for feature extraction
###############################################################################


namelist=['grad'+str(i) for i in range(13)] 
beta_spd = 0.0007
# beta_spd = 1000000000
beta_m = 0.0001
para1=1919
para2 = [0.9*beta_spd,9.8,0.85,0.05,0.9*beta_m]

###############################################################################
### Proposed data augmentation method applications
###############################################################################

travelpath=r''  # Enter the address of travel data
                # For instance: travelpath=r'D:\Tesla model 3\train\original' 

trainsavepath=r''
valsavepath=r''

proposed(travelpath,trainsavepath,valsavepath)


###############################################################################
### Baseline1 data augmentation method applications
###############################################################################


save_baseline_1=r''
readtrain=r''
readval=r''
baseline_1(readtrain,readval,save_baseline_1)


###############################################################################
### Baseline2 data augmentation method applications
###############################################################################


input_folder = r''
output_folder_train = r''
output_folder_val = r''
random_select_files(input_folder, output_folder_train, output_folder_val, percentage=20)
baseline2(output_folder_train)



###############################################################################
### Five fold cross validation
###############################################################################
train_path=r''
val_path=r''


lgb1_val=validation(train_path,val_path,'LGB_1')
lgb2_val=validation(train_path,val_path,'LGB_2')
lgb2_val=validation(train_path,val_path,'XGBoost')


```


## Disclaimer
This software is provided as freeware, intended solely for non-commercial, educational, and research purposes. It must not be used for any commercial purposes without prior authorization from the software developer. Any use for commercial purposes without such authorization will render you and the users responsible for any resultant liabilities, and the software developer and the platform will not be held responsible for any consequences arising therefrom.
Users assume all risks associated with the use of this software. The developer and associated platforms disclaim any liability for special, incidental, direct, or indirect damages arising out of or in connection with the use or inability to use the software. This includes, but is not limited to, any loss of data or property, and any resulting or related liabilities to the user or any third parties.
By downloading or using this code, you signify your agreement to these terms.

## Contact Us

Please contact us if you need further technical support. If you have any trouble with this repo, feel free to contact us by e-mail. We'll try to resolve the issue as soon as possible! Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.\
Email contact: &nbsp; [MOTIVES Lab](mailto:motives.lab@gmail.com), &nbsp; [Yifan Ma](mailto:jlumayf@163.com).