# Online-Kernel-Learning
This is all the Matlab codes used in "Large Scale Online Kernel Learning"

Research Paper available at http://jmlr.org/papers/v17/14-148.html

Authors: Lu Jing, Steven Hoi

Contact: chhoi@ntu.edu.sg, jing.lu.2014@phdis.smu.edu.sg

This package, in Matlab, includes the most widely used online kernel learning algorithms for binary classification, multiple kernel classification and regression. This is designed for machine learning researcher who are interested in Matlab coding and is very easy to understand. For high efficient implementations, please refer to our c++ toolbox:  https://github.com/LIBOL/KOL


The algorithms in this package includes:

1. Perceptron: The kernelized Perceptron without budget maintainance. http://cseweb.ucsd.edu/~yfreund/papers/LargeMarginsUsingPerceptron.pdf

2. Online Gradient Descent (OGD): The kernelized online gradient descent algorithm without budget maintainance. 
http://eprints.pascal-network.org/archive/00002055/01/KivSmoWil04.pdf

3. Random Budget Perceptron (RBP): Budgeted perceptron algorithm with random support vector removal strategy. 
 http://air.unimi.it/bitstream/2434/26350/1/J29.pdf

4. Forgetron: Forgetron algorithm that maintains the budget size by discarding the oldest support vectors. 
http://papers.nips.cc/paper/2806-the-forgetron-a-kernel-based-perceptron-on-a-fixed-budget.pdf

5. Projectron: The Projectron algorithm using budget projection strategy. 
http://eprints.pascal-network.org/archive/00004472/01/355.pdf

6. Projectron++: The aggressive version of Projectron algorithm that updates with both margin error and mistake case. 
http://eprints.pascal-network.org/archive/00004472/01/355.pdf

7. BPAs: The budget passive-aggressive algrotihtm with simple supprot removal strategy.
http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_WangV10.pdf

8. BOGD: The budget online gradient descent algorithm by SV removal strategy 
http://arxiv.org/ftp/arxiv/papers/1206/1206.4633.pdf

9. FOGD: The Fourier Online Gradient Descent algorithm using functional approximation method.
http://jingonline.weebly.com/uploads/5/3/7/3/53733905/lu15a.pdf

10. NOGD: The Nystrom Online Gradient Descent algorithm using functional approximation method.[pdf]
http://jingonline.weebly.com/uploads/5/3/7/3/53733905/lu15a.pdf

The last two were proposed by our group and published on Journal of Machine Learning Research. If you need to use this code package, please cite our paper as: 
________________________________________

Lu J, Hoi S C H, Wang J, et al. Large scale online kernel learning[J]. Journal of Machine Learning Research, 2016, 17(47): 1.

or bib:
________________________________________
@article{lu2016large,
  title={Large scale online kernel learning},
  author={Lu, Jing and Hoi, Steven CH and Wang, Jialei and Zhao, Peilin and Liu, Zhi-Yong},
  journal={Journal of Machine Learning Research},
  volume={17},
  number={47},
  pages={1},
  year={2016},
  publisher={Journal of Machine Learning Research/Microtome Publishing}
}
___________________________________________

Related links:

Our C++ toolbox for online kernel learning: https://github.com/LIBOL/KOL

Steven Hoi's home page: http://stevenhoi.org/

LU Jing's home page: http://jingonline.weebly.com/

LIBOL: http://libol.stevenhoi.org/

LIBSOL: http://libsol.stevenhoi.org/

Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page

LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

Journal of Machine Learning Reseaerch: http://jmlr.org/papers/v17/14-148.html


Our Matlab codes for all experiments in the research paper:

Our follow-up research in online multiple kernel learning: 


A follow-up work to our proposed algorithm in NIPS: https://papers.nips.cc/paper/6560-dual-space-gradient-descent-for-online-learning.pdf
