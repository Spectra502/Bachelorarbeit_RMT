from avalanche.training.determinism.rng_manager import RNGManager

def run_training(train_stream, test_stream, cl_strategy, cl_strategy_Naive_Pre, eval_plugin):
    RNGManager.set_random_seeds(1234)
    print("TRAINING THE MODEL WITH FINETUNING FOR THE FIRST PARAMETERGROUP\n")
    # Create empty lists for the results
    train_results_list_NAIVE = []
    test_results_list_NAIVE = []
    train_results_list_strategy = []
    test_results_list_strategy = []
    
    # Train and test loop (Naive Finetuning: Parametergroup 1)
    experience = train_stream[0]
    print("### Start training the model on experience", experience.current_experience,"###")
    train_results_Naive_Pre = cl_strategy_Naive_Pre.train(experience)
    train_results_list_NAIVE.append(train_results_Naive_Pre)
    train_results_list_strategy.append(train_results_Naive_Pre)
    print("### Training the model on experience", experience.current_experience, "completed ###\n")
        
    print("### Start evaluating the on experience", experience.current_experience,"trained model ###")
    test_results_Naive_Pre = cl_strategy_Naive_Pre.eval(test_stream)
    test_results_list_NAIVE.append(test_results_Naive_Pre)
    test_results_list_strategy.append(test_results_Naive_Pre)
    print("### Evaluating the on experience", experience.current_experience, "trained model completed ###\n")

    # Output final results
    #print("length of train results list:", len(train_results_list_NAIVE), "; train results list: ", train_results_list_NAIVE)
    #print("length of test results list:", len(test_results_list_NAIVE), "; test results list: ", test_results_list_NAIVE)

    
    # Train and test loop (DER: Parametergruppe 2 - 7)
    for experience in train_stream[1:7]:
        print("### Start training the model on experience", experience.current_experience,"###")
        train_results_strategy = cl_strategy.train(experience)
        train_results_list_strategy.append(train_results_strategy)
        print("### Training the model on experience", experience.current_experience, "completed ###\n")
        
        print("### Start evaluating the on experience", experience.current_experience,"trained model ###")
        test_results_strategy = cl_strategy.eval(test_stream)
        test_results_list_strategy.append(test_results_strategy)
        print("### Evaluating the on experience", experience.current_experience, "trained model completed ###\n")
    
        # Output final results
        #print("length of train results list:", len(train_results_list_strategy), "; train results list: ", train_results_list_strategy)
        #print("length of test results list:", len(test_results_list_strategy), "; test results list: ", test_results_list_strategy)
    metric_dict = eval_plugin.get_all_metrics()
    return train_results_list_NAIVE, test_results_list_NAIVE, train_results_list_strategy, test_results_list_strategy, metric_dict
    