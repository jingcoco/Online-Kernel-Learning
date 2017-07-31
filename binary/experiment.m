function demo(dataset_name,data)
% Experiment_OL_K_M: the main procedure evaluating all the algorithm on the same dataset
%--------------------------------------------------------------------------
% Input:
%      dataset_name: the name of the dataset file
% Output:
%      a table of the final mistake rates, time consumption for all the algorithms
%--------------------------------------------------------------------------

%% load dataset
load(sprintf('data/%s',dataset_name));
[n,d]       = size(data);
runs = 20;
%% set parameters
options.Budget=50;
options.eta    =  0.2;
options.lambda =  1/(n^2);
options.gamma  =  10;
options.t_tick = round(n/15);
options.sigma  =64;
%options for bpas
options.C_BPAs = 1;
%options for NOGD
options.eta_nogd=0.2;
options.k = 0.2;
%options for FOGD
options.eta_fou=0.002;
options.D=4;

%%  choose a subset of the whole dataset (default: using all)

Y = data(1:n,1);
X = data(1:n,2:d);

    %% run experiments:

      for i=1:20,
        ID = ID_all(i,:); 
        i
        %1. perceptron
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = perceptron(Y,X,options,ID);
        nSV_PE(i)             = length(classifier.SV);
        err_PE(i)             = err_count;
        time_PE(i)            = run_time;
        mistakes_list_PE(i,:) = mistakes;
        SVs_PE(i,:)           = SVs;
        TMs_PE(i,:)           = TMs;
        

        %1. ogd
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = OGD(Y,X,options,ID);
        nSV_OGD(i)             = length(classifier.SV);
        err_OGD(i)             = err_count;
        time_OGD(i)            = run_time;
        mistakes_list_OGD(i,:) = mistakes;
        SVs_OGD(i,:)           = SVs;
        TMs_OGD(i,:)           = TMs;
      
        
        %2. RBP
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = RBP(Y,X,options,ID);
        nSV_RP(i)             = length(classifier.SV);
        err_RP(i)             = err_count;
        time_RP(i)            = run_time;
        mistakes_list_RP(i,:) = mistakes;
        SVs_RP(i,:)           = SVs;
        TMs_RP(i,:)           = TMs;
        
        
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = forgetron(Y,X,options,ID);
        nSV_FP(i)             = length(classifier.SV);
        err_FP(i)             = err_count;
        time_FP(i)            = run_time;
        mistakes_list_FP(i,:) = mistakes;
        SVs_FP(i,:)           = SVs;
        TMs_FP(i,:)           = TMs;
        
         %3. projectron
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = projectron(Y,X,options,ID);
        nSV_PJ(i)             = length(classifier.SV);
        err_PJ(i)             = err_count;
        time_PJ(i)            = run_time;
        mistakes_list_PJ(i,:) = mistakes;
        SVs_PJ(i,:)           = SVs;
        TMs_PJ(i,:)           = TMs;

        %4. projectron++
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = proplus(Y,X,options,ID);
        nSV_PP(i)             = length(classifier.SV);
        err_PP(i)             = err_count;
        time_PP(i)            = run_time;
        mistakes_list_PP(i,:) = mistakes;
        SVs_PP(i,:)           = SVs;
        TMs_PP(i,:)           = TMs;
        
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = BOGD(Y,X,options,ID);
        nSV_BO(i)             = length(classifier.SV);
        err_BO(i)             = err_count;
        time_BO(i)            = run_time;
        mistakes_list_BO(i,:) = mistakes;
        SVs_BO(i,:)           = SVs;
        TMs_BO(i,:)           = TMs;
        

%                 
        %4. BPAs
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = BPAs(Y,X,options,ID);
        nSV_BPAs(i)             = length(classifier.SV);
        err_BPAs(i)             = err_count;
        time_BPAs(i)            = run_time;
        mistakes_list_BPAs(i,:) = mistakes;
        SVs_BPAs(i,:)           = SVs;
        TMs_BPAs(i,:)           = TMs;
 
               
                %5. FoGD
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = FouGD(Y,X,options,ID);
        nSV_FGD(i)             = length(classifier.SV);
        err_FGD(i)             = err_count;
        time_FGD(i)            = run_time;
        mistakes_list_FGD(i,:) = mistakes;
        SVs_FGD(i,:)           = SVs;
        TMs_FGD(i,:)           = TMs;

        %6. NysGD
        [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = NysGD(Y,X,options,ID);
        nSV_NGD(i)             = length(classifier.SV);
        err_NGD(i)             = err_count;
        time_NGD(i)            = run_time;
        mistakes_list_NGD(i,:) = mistakes;
        SVs_NGD(i,:)           = SVs;
        TMs_NGD(i,:)           = TMs;
        
    end
    fprintf(1,'-------------------------------------------------------------------------------\n');
     fprintf('Algorithm : Mistake Rates, size of support vectors, cpu running time\n');
    fprintf(1,'Perceptron   &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t \\\\\n', mean(err_PE)/n*100, std(err_PE)/n*100, mean(time_PE));
     fprintf(1,'OGD          &%.3f \t\\%%$\\pm$ %.3f \t& %.3f \t \\\\\n', mean(err_OGD)/n*100, std(err_OGD)/n*100, mean(time_OGD));
     fprintf(1,'RBP          &%.3f \t\\%%$\\pm$ %.3f \t&  %.3f \t \\\\\n', mean(err_RP)/n*100, std(err_RP)/n*100,mean(time_RP));
    fprintf(1,'Forgetron    &%.3f \t\\%%$\\pm$ %.3f \t&  %.3f \t \\\\\n', mean(err_FP)/n*100, std(err_FP)/n*100, mean(time_FP));
    fprintf(1,'Projectron   &%.3f \t\\%%$\\pm$ %.3f \t&  %.3f \t \\\\\n', mean(err_PJ)/n*100, std(err_PJ)/n*100, mean(time_PJ));
    fprintf(1,'Projectron++ &%.3f \t\\%%$\\pm$ %.3f \t&  %.3f \t \\\\\n', mean(err_PP)/n*100, std(err_PP)/n*100, mean(time_PP));
    fprintf(1,'BPAs         &%.3f \t\\%%$\\pm$ %.3f \t&  %.3f \t \\\\\n', mean(err_BPAs)/n*100, std(err_BPAs)/n*100, mean(time_BPAs));
    fprintf(1,'BOGD         &%.3f \t\\%%$\\pm$ %.3f \t&  %.3f \t \\\\\n', mean(err_BO)/n*100, std(err_BO)/n*100,mean(time_BO));
    fprintf(1,'FOGD        &%.3f \t\\%%$\\pm$ %.3f \t&  %.3f \t \\\\\n', mean(err_FGD)/n*100, std(err_FGD)/n*100, mean(time_FGD));
    fprintf(1,'NOGD        &%.3f \t\\%%$\\pm$ %.3f \t&  %.3f \t \\\\\n', mean(err_NGD)/n*100, std(err_NGD)/n*100,  mean(time_NGD));
    fprintf(1,'-------------------------------------------------------------------------------\n');

  
figure
grid on
hold on
box on
mean_mistakes_OGD = mean(mistakes_list_OGD);
plot(mistakes_idx, mean_mistakes_OGD,'m-<');
legend('OGD_{avg}');
xlabel('Number of samples');
ylabel('Online mistake rate')
