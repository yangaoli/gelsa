GeLSA: a GPU-accelerated Local Similarity Analysis Tool
=========================================================

INSTRODUCTION
--------------

GeLSA (GPU-accelerated extended Local Similarity Analysis) this novel multi-core accelerated computing tool enables local similarity analysis (LSA) for large-scale time series data in microbiome and environmental sciences. Compared to the previous most efficient LSA implementation (eLSA), GeLSA achieved approximately a 144-fold increase in computational efficiency on GPU machines.This is because GeLSA adapted the max sum subarray dynamical programming algorithm for LSA, allowing efficient core-level parallelisation to use modern CPU/GPU architectures. GeLSA also generally accelerates LSA-derived algorithms, including the local trend analysis (LTA), permutation-based MBBLSA, theory-based DDLSA and STLTA methods.As demonstrated by benchmarks, GeLSA maintained the accuracy of those methods while substantially improving their efficiency. 


METHODS
-------------

<img src="./images/fig2.jpg" alt="fig2.jpg" width="450" height="450" />

Figure 1. The analysis workflow of Local Similarity Analysis (LSA) tools. Users start with raw data (matrices of time series) as input and specify their requirements as parameters. The LSA tools subsequently F-transform and normalize the raw data and then calculate the Local Similarity (LS) Scores and the Pearson’s Correlation Coefficients. The tools then assess the statistical significance (P-values) of these correlation statistics using permutation test and filter out insignificant results. Finally, the tools construct a partially directed association network from significant associations.

<img src="./images/fig4.jpg" alt="fig4.jpg" width="450" height="550" />

Figure 2. Liquid Association / Mediated correlation and example Cytoscape diagrams for all liquid association types of factors X, Y and Z: (A) High Z level enhances the positive correlation between X and Y; (B) Low Z level enhances the negative correlation between X and Y; (C) Low Z level enhances the positive correlation between X and Y; (D) High Z level enhances the negative correlation between X and Y. And (E) A flowchart for incorporating Liquid Association (LA) with Local Similarity (LS) Analysis (LSA). First we use LSA to find candidate local and without time-delayed associations between factors X and Y. The results were filtered based on p-values, q-values and effect (LS score). Then, given the significant LSA factors X and Y, we compute LA score to scout any environmental/OTU factors to discover potential mediating factor Z. Next, a permutation test for liquid association is performed and the results were filtered based on p-values, q-values and effect (LA score) to remove insignificant triplets. Finally, we use the software Cytoscape to visualize the results.


INSTALL
-----------------
(1). # for use #

Currently, the package is maintained only for Linux (Ubuntu) due to compilation requirements for the core computational components.

Firstly, in Linux install the prerequisites: C++ (build-essential), Python(dev).

Then, download and unzip the latest main branch to the gelsa folder.

        gelsa>  bash CPU_command.sh                                                                                     # make lsa package and computate by using cpu
        gelsa>  python in_out_data.py &&  lsa_compute test.txt result -d 10 -r 1 -s 50 -p theo -T 0.1                   # a test script is available

        gelsa>  bash GPU_command.sh                                                                                     # To create an LSA package that can utilize GPU acceleration when available(Ubuntu 22.04 Required)
        gelsa>  python in_out_data.py &&  lsa_compute test.txt result -d 10 -r 1 -s 50 -p theo -T 0.1                   # a test script is available


(2). # for development #

GeLSA is open source and your contributions are greatly welcome.

First, use git to fork a copy of gelsa on github.com:
        
        gelsa> git clone ssh://git@github.com/your/gelsa gelsa

Then, make your edits and create a pull request to merge back.


EXECUTABLES
--------------------
The following executable will be available from your python scripts directory (typically already in $PATH).

    lsa_compute                       # for LSA/MBBLSA/DDLSA/LTA/STLTA/PERMUTAION  computation

NOTES
----------------------    
The lsa and lta computation capacities (lsa/lta/permutation) of eLSA and new computation capacities (ddlsa/bblsa/stlta) are available through GeLSA

CONTACT
----------------------
lcxia at scut dot edu dot cn

CITATIONS
----------------------
Please cite the references 1 and 2 if any part of the ELSA python package was used in your study.Please also cite 3 if local trend analysis (LTA) was used in your study. Please also cite the reference 4 and 5 if you used the old LSA R script, which is no loger maintained. Please also cite 6 if Moving Block Bootstrap LSA (MBBLSA) was used in your study. Please also cite 7 if Data-Driven LSA (DDLSA) was used in your study. Please also cite 8 if Steady-state Theory Local Trend Analysis (STLTA) was used in your study. 


1. Li C Xia, Dongmei Ai, Jacob Cram, Jed A Fuhrman, Fengzhu Sun. Efficient Statistical Significance Approximation for Local Association Analysis of High-Throughput Time Series Data. Bioinformatics 2013, 29(2):230-237. (https://doi.org/10.1093/bioinformatics/bts668)
2. Li C Xia, Joshua A Steele, Jacob A Cram, Zoe G Cardon, Sheri L Simmons, Joseph J Vallino, Jed A Fuhrman and Fengzhu Sun. Extended local similarity analysis (eLSA) of microbial community and other time series data with replicates. BMC Systems Biology 2011, 5(S2):S15 (https://doi.org/10.1186/1752-0509-5-S2-S15)
3. Li C Xia, Dongmei Ai, Jacob Cram, Xiaoyi Liang, Jed Fuhrman, Fengzhu Sun. Statistical significance approximation in local trend analysis of high-throughput time-series data using the theory of Markov chains. BMC Bioinformatics 2015, 16, 301 (https://doi.org/10.1186/s12859-015-0732-8)
4. Joshua A Steele, Peter D Countway, Li Xia, Patrick D Vigil, J Michael Beman, Diane Y Kim, Cheryl-Emiliane T Chow, Rohan Sachdeva, Adriane C Jones, Michael S Schwalbach, Julie M Rose, Ian Hewson, Anand Patel, Fengzhu Sun, David A Caron, Jed A Fuhrman. Marine bacterial, archaeal and protistan association networks reveal ecological linkages The ISME Journal 2011, 51414–1425
5. Quansong Ruan, Debojyoti Dutta, Michael S. Schwalbach, Joshua A. Steele, Jed A. Fuhrman and Fengzhu Sun Local similarity analysis reveals unique associations among marine bacterioplankton species and environmental factors Bioinformatics 2006, 22(20):2532-2538
6. Zhang F, Shan A, Luan Y. A novel method to accurately calculate statistical significance of local similarity analysis for high-throughput time series. Stat Appl Genet Mol Biol 2018; 17:20180019. 
7. Zhang F, Sun F, Luan Y. Statistical significance approximation for local similarity analysis of dependent time series data. BMC Bioinformatics 2019;20:53. 
8. Shan A, Zhang F, Luan Y. Efficient approximation of statistical significance in local trend analysis of dependent time series. Front Genet 2022;13:729011
