#Entropy calculations
#Thomas Parmer, 2022

from scipy.stats import entropy


#entropy function of possible configurations
def config_entropy(diffusion,nodes=None,base=2,binary=True,normalized=True,strict=False):
    """ determine the entropy of an iterable keyed as {timestep: {node: activation_probabilities} }
    to determine information gain from reducing possible network configurations, 
    normalization based on total possible entropy and total possible network configurations,
    strict only reduces configurations based on constants rather than probabilities """
    
    if not diffusion or not diffusion[0]: #no diffusion to measure
        return 0.0
    if not nodes:
        nodes=diffusion[0].keys()
    config_entropy={t:0.0 for t in diffusion}
    configs={t:1.0 for t in diffusion}
    #max possible entropy
    if binary:
        max_entropy=sum([entropy([.5,.5],base=base) for node in nodes])
        max_configs=2**len(nodes)
    #print max_entropy,"{:e}".format(max_configs)
    for t in diffusion:
        for node in nodes:
            config_entropy[t]+=entropy([diffusion[t][node],1-diffusion[t][node]],base=base)
            if strict and diffusion[t][node]<1 and diffusion[t][node]>0: #non-constant so consider both possibilities
                configs[t]*=2
            else:
                configs[t]*=1/max([diffusion[t][node],1-diffusion[t][node]])
        
    if normalized:
        return {t:config_entropy[t]/max_entropy for t in config_entropy},{t:configs[t]/max_configs for t in configs}
    return config_entropy,configs


#entropy function of possible configurations per node
def config_node_entropy(diffusion,nodes=None,base=2,binary=True,normalized=True,strict=False):
    """ determine the entropy of an iterable keyed as {timestep: {node: activation_probabilities} }
    to determine information gain from reducing possible network configurations, 
    normalization based on total possible entropy and total possible network configurations,
    strict only reduces configurations based on constants rather than probabilities """
    
    if not diffusion or not diffusion[0]: #no diffusion to measure
        return 0.0
    if not nodes:
        nodes=diffusion[0].keys()
    config_entropy={node:{t: 0.0 for t in diffusion} for node in  nodes}
    configs={node:{t: 1.0 for t in diffusion} for node in nodes}
    #max possible entropy
    if binary:
        max_entropy=entropy([.5,.5],base=base) #sum([entropy([.5,.5],base=base) for node in nodes])
        max_configs=2 #2**len(nodes)
    #print max_entropy,"{:e}".format(max_configs)
    for t in diffusion:
        for node in nodes:
            config_entropy[node][t]=entropy([diffusion[t][node],1-diffusion[t][node]],base=base)
            if strict and diffusion[t][node]<1 and diffusion[t][node]>0: #non-constant so consider both possibilities
                configs[node][t]=2
            elif strict:
                configs[node][t]=1 #1/max([diffusion[t][node],1-diffusion[t][node]])
            else:
                configs[node][t]=base**entropy([diffusion[t][node],1-diffusion[t][node]],base=base)
        
    if normalized:
        config_entropy={node:{t: config_entropy[node][t]/max_entropy for t in diffusion} for node in nodes}
        configs={node:{t: configs[node][t]/max_configs for t in diffusion} for node in nodes}
        
    return config_entropy,configs       


#find all seed entropies over time
def seed_entropies(modules,seeds,nodes=None,base=2,binary=True,normalized=True,data=True,strict=False):
    """ determine the entropy of an iterable keyed as {timestep: {node: activation_probabilities} } for given nodes
    to determine information gain from reducing possible network configurations for all seeds from modules """
    
    seed_entropies,seed_configs={},{}
    for seed in seeds:
        ce=config_entropy(modules[seed],nodes=nodes,base=base,binary=binary,normalized=normalized,strict=strict)
        seed_entropies[seed],seed_configs[seed]=ce[0],ce[1]
        
    if data:
        #assume all seeds have the same number of iterations
        print("average entropy:",[np.mean([seed_entropies[seed][t] for seed in seed_entropies]) for t in seed_entropies[seed]])
        print("average configurations:",[np.mean([seed_configs[seed][t] for seed in seed_configs]) for t in seed_configs[seed]])
        
    return seed_entropies,seed_configs


#find all seed entropies over time per node
def seed_node_entropies(modules,seeds,nodes=None,base=2,binary=True,normalized=True,data=True,strict=False):
    """ determine the entropy of an iterable keyed as {timestep: {node: activation_probabilities} } for given nodes
    to determine information gain from reducing possible network configurations for all seeds from modules """
    
    seed_entropies,seed_configs={},{}
    for seed in seeds:
        ce=config_node_entropy(modules[seed],nodes=nodes,base=base,binary=binary,normalized=normalized,strict=strict)
        seed_entropies[seed],seed_configs[seed]=ce[0],ce[1]
        
    if data:
        #assume all seeds have the same number of iterations
        print("average entropy, average configurations")
        for node in seed_entropies[seed]:
            print(node,[np.mean([seed_entropies[seed][node][t] for seed in seed_entropies]) for t in seed_entropies[seed][node]])
            print(node,[np.mean([seed_configs[seed][node][t] for seed in seed_configs]) for t in seed_configs[seed][node]])
        
    return seed_entropies,seed_configs


#compare entropies across different update schedules
def compare_entropies_asynchronous(N,seeds,sunits,sunit_map,runs=100,tau=0.0,iterations=10,unknown_prob=0.5,reduced=False,ds=None,
    stats={},pinning={},results=False,schedule='synchronous',models=100,order=None,verbose=False):
    """ possible udpate schedules are 'synchronous', 'deterministic asynchronous', 'stochastic without replacement', 'stochastic with replacement', 'block udpate' 
    NOTE: this can only accept seeds of the same length; order argument ignored unless update='block update' """
    
    entropy_modules,entropy_sim={seed:[0.0 for i in range(iterations+1)] for seed in seeds},{seed:[0.0 for i in range(iterations+1)] for seed in seeds}
    modules,translator={},{}
    time_limit=iterations+2
    act_prob_sim={seed:{} for seed in seeds}
    simulations={}
    for seed in seeds:
        length=len(eval(seed)) #seed expected to be a string!
        if verbose: print(seed)
        #get deterministic modules and updates
        if schedule=='synchronous': update='synchronous'
        if schedule=='block update': update='asynchronous'
        if schedule=='synchronous' or schedule=='block update':
            modules,translator=find_modules(N,length,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,iterations=iterations,
                                  data=True,seeds=[eval(seed)],update=update,order=order,regenerate=False,verbose=False)
            simulations=compare_simulations(N,[seed],sunit_map,modules,translator,length=length,runs=runs,tau=tau,iterations=iterations,
             unknown_prob=unknown_prob,stats={},results=results,time_limit=time_limit,update=update,order=order,regenerate=False)
            aggregate_simulation(seed,simulations,act_prob_sim)
            #print act_prob_sim[seed][iterations]          
            #calculate entropy for modules
            seed_entropy,seed_configs=config_entropy(modules[seed],base=2,normalized=True)
            entropy_modules[seed]=[seed_entropy[x] for x in seed_entropy]
            #calculate entropy for simulations
            seed_entropy,seed_configs=config_entropy(act_prob_sim[seed],base=2,normalized=True) 
            entropy_sim[seed]=[seed_entropy[x] for x in seed_entropy]

        else: #get stochastic modules and updates
            deterministic=False
            replacement=False
            if schedule=='deterministic asynchronous':
                deterministic=True
            if schedule=='stochastic with replacement':
                replacement=True
            d,s=compare_simulations_asynchronous(N,[seed],sunit_map,length=length,runs=runs,tau=tau,iterations=iterations,unknown_prob=unknown_prob,
                reduced=reduced,ds=ds,stats={},results=results,time_limit=time_limit,deterministic=deterministic,replacement=replacement,models=models,verbose=verbose)
            for m in range(len(d)):
                modules[seed]=d[m][seed]
                act_prob_sim[seed]={}
                aggregate_simulation(seed,s[m],act_prob_sim)
                #print 'Model:',m,len(modules[seed]),len(act_prob_sim[seed]),len(entropy_modules[seed])
                #calculate entropy for modules
                seed_entropy,seed_configs=config_entropy(modules[seed],base=2,normalized=True)
                entropy_modules[seed]=[entropy_modules[seed][i]+[seed_entropy[x] for x in seed_entropy][i] for i in range(iterations+1)]
                #calculate entropy for simulations
                seed_entropy,seed_configs=config_entropy(act_prob_sim[seed],base=2,normalized=True)
                entropy_sim[seed]=[entropy_sim[seed][i]+[seed_entropy[x] for x in seed_entropy][i] for i in range(iterations+1)]
                #print 'Entropy:',[seed_entropy[x] for x in seed_entropy]
            entropy_modules[seed]=[entropy_modules[seed][i]/len(d) for i in range(iterations+1)]
            entropy_sim[seed]=[entropy_sim[seed][i]/len(d) for i in range(iterations+1)]
    
    return entropy_modules,entropy_sim
