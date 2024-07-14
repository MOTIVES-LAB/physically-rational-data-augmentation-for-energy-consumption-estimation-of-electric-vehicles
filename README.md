# Physically Rational Data Augmentation for Energy Consumption Estimation of Electric Vehicles
Data and Code for the paper **Physically rational data augmentation for energy consumption estimation of electric vehicles**.
## Overview
The physically rational data augmentation and validation is a Python-based program designed to realize the data augmentation of the vehicle driving data in real world. This tool through sample split and sample concatenation of any driving data achieved data augmentation and verify the effectiveness of the proposed method in existing energy consumption estimation models.

<p align="center">
  <img src="https://github.com/MOTIVES-LAB/physically-rational-data-augmentation-for-energy-consumption-estimation-of-electric-vehicles/blob/main/graphical%20abstract.png" alt="drawing" width="1200"/>
</p>  

## Features
- **Dataset**: High quality real-world driving data of Tesla Model 3;
- **Data augmentation approach**: Using sub-trips (mirco-trips) for more effecitve (physical rational) data augmentation;
- **Validation**: Using small-sized training set (20%) for testing the generalizability of the model performance.

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
## References
- Please cite the following paper when referring to this approach:  
`
Ma, Yifan, et al. "Physically rational data augmentation for energy consumption estimation of electric vehicles." Applied energy，accepted (2024).  
`\
Will be available soon at: [Applied Energy](https://doi.org/10.1016/j.apenergy.2024.123871)  

## Disclaimer
This software is provided as freeware, intended solely for non-commercial, educational, and research purposes. It must not be used for any commercial purposes without prior authorization from the software developer. Any use for commercial purposes without such authorization will render you and the users responsible for any resultant liabilities, and the software developer and the platform will not be held responsible for any consequences arising therefrom.
Users assume all risks associated with the use of this software. The developer and associated platforms disclaim any liability for special, incidental, direct, or indirect damages arising out of or in connection with the use or inability to use the software. This includes, but is not limited to, any loss of data or property, and any resulting or related liabilities to the user or any third parties.
By downloading or using this code, you signify your agreement to these terms.

## Contact Us

Please contact us if you need further technical support. If you have any trouble with this repo, feel free to contact us by e-mail. We'll try to resolve the issue as soon as possible! Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.\
Email contact: &nbsp; [MOTIVES Lab](mailto:motives.lab@gmail.com), &nbsp; [Yifan Ma](mailto:jlumayf@163.com).

<p align="center">
  <img src="https://github.com/MOTIVES-LAB/generalized-energy-consumption-evaluation-for-ev/blob/main/figures/new_logo_trans.png" alt="drawing" width="200"/>
</p>  
