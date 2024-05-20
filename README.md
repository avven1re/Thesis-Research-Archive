# Thesis-Research-Archive
Github: https://github.com/avven1re/Thesis-Research-Archive
## Introduction
This is the research archive of my thesis: *Selection Bias Correction: Optimizing Estimation by Machine Learning Techniques and Doubly Robust Method in Pseudo-Weighting Approach*.

A project that compares several classification methods under Liu's Framework and its expanded version with doubly robust implementation. Here is the brief view of the structure of the Thesis-Research-Archive:

```
Thesis-Research-Archive
├── README.md
├── Requirements.md
├── 📁Liu's_Framework
│   ├── 📁Data
│   │   ├── DataGeneration.jl
│   │   └── README.txt
│   └── 📁scr
│   │   └── 📁functions
│   │   └── Restructured_LiuFramework.jl
│   │   └── [Example Code]Restructured_LiuFramework.R
|   └──📁output
└── 📁Liu's_Framework&DoublyRobust
|   ├── 📁Data
|   |   ├── DataGeneration.jl
|   |   └── README.txt
│   └── 📁scr
│   │   └── 📁functions
│   │   └── DoublyRobust.jl
│   │   └── [Example Code]Doubly Robust.R
|   └──📁output
```
`Requirements.md` contains the information of software and packages of use.
And there are two simulation study in two separate folders:

 - 📁Liu's_Framework,
 - 📁Liu's_Framework&DoublyRobust.

In each folder of simulation study, there are three folders including:

 - 📁**Data**: Contains the file of generating function for the simulated data and an .txt file for both simulated data and the DOKR data.
 - 📁**scr**: Contains the source code of the simulation study.
 - 📁**output**: Contains the outputs.

 There are two datasets in this project: 

- Simulated data,
- the registered data of Dutch Online Kilometer Registration (DOKR data).

Note that *only* the simulated data is available publicly in this research archive. The registered data of Dutch Online Kilometer Registration (DOKR) data is stored in CBS and following the protocol of CBS. For more information of the datasets, please check the **Data** folder in each simulation study folder. **FETC-approved: 23–1784 and 23–1842**

## A Guide of using the files in **scr** folder:
Each data were conducted under two different programming languages:

- the Simulated data: `Julia`
- DOKR data: `R`

 Due to the DOKR data not being publicly available, only the example code for the simulation study from this dataset is provided (`[Example Code] .R`). It is not reproducible unless you have access to the DOKR data, and the output files are not extractable from the CBS environment.

In each **scr** folder, you will find a **functions** folder, a `[Example Code] .R` file, and a `.jl` file (`Restructured_LiuFramework.jl` or `DoublyRobust.jl`). The `.jl` file includes the data generation (simulated data) and the whole simulation study process by calling functions in **functions** folder and the **Data** folder (`DataGeneration.jl`). Please check the `Requirements.md` to set up the Julia environment first, and then you are able to reproduce the two simulation study of simulated data by simply executing the `.jl` file in **scr** folder (`Restructured_LiuFramework.jl` and `DoublyRobust.jl`). The results (relative bias and RMSEs of each classification method) will be stored as `.csv` in the **output** folder.

All simulation processes are seeded.


## Permission and Access
This research archive is available on [Github](https://github.com/avven1re/Thesis-Research-Archive) only with the simulation study conducting under the simulated data.


