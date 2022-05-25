#Simulations of Boolean networks
#Thomas Parmer, 2022

from utils import *
import random
from brute_force_computations import run_network_dynamics
from modules import *

#create subsample of modules
def create_samples(modules,size,sunit_map,translator,s=1):
    subsample={}
    while len(subsample)<size:
        key=random.choice(modules.keys())
        #check that it is a valid length and does not have a contradiction
        sunits=to_list(key,sunit_map,translator)
        if len(sunits)!=s:
            continue
        if len({sunit[:-2] for sunit in sunits})!=len(sunits): #there is a contradiction
            continue
        subsample[key]=modules[key]
        
    return subsample


#print stats in a human-readable way
def print_stats(stats,total=1):
    
    print('total seeds: '+str(total))
    print('true positives: '+str(stats['tps']))
    print('false negatives: '+str(stats['fns']))
    print('true negatives: '+str(stats['tns']))
    print('false positives: '+str(stats['fps']))
    print('accuracy: '+str(stats['accuracy']))
    print('recall: '+str(stats['recall']))
    print('precision: '+str(stats['precision']))
    print('average similarity: '+str(stats['avg_sim']))
    print('errors: '+str(stats['errors']))


#updated to work with different update schedules
def create_simulations(ND,seeds,dsunit_map,dtranslator,length=1,runs=100,iterations=10,unknown_prob=0.5,time_limit=12,
                       act_prob_sim={},simulations={},update='synchronous',order=None,regenerate=False):
    """ runs various simulations of the seeds, returns the generated simulations 
    iterations refers to the number of iterations past the initial configuration at t=0 (iterations+1 total)
    NOTE: this applies only to Boolean networks with calculable DCMs and assumes steady-state perturbation """
    
    #act_prob_sim={}
    #simulations={} #store the random simulations per seed
    iterations=iterations+1 #include 0
    #DCM=ND.dynamics_canalization_map(output=None, )
    nodes={node.name for node in ND.nodes}
    #make sure all seeds are in the translator (if no seeds are given, it will find all possible seeds of length s)
    dtranslator=create_translator(ND,s=length,sunits=None,sunit_map=dsunit_map,translator=dtranslator,seeds=seeds,samples=None)
    for seed in seeds:
        sunits=to_list(str(seed),dsunit_map,dtranslator)
        if len(sunits)!=length:
            continue
        if len({sunit[:-2] for sunit in sunits})!=len(sunits): #there is a contradiction
            continue
        act_prob_sim[seed] = {i:{node: 0.0 for node in nodes} for i in range(iterations)}
        simulations[seed]={}

        #get the ground truth for the seed based on several runs
        for i in range(runs):
            #initiate a random condition
            alt_seed=[sunit for sunit in sunits]
            for node in nodes-{sunit[:-2] for sunit in alt_seed}:
                if random.random()<unknown_prob:
                    alt_seed.append(node+'-1')
                else:
                    alt_seed.append(node+'-0')
            #find the information diffusion for the sample based on the BFS threshold algorithm
            #print alt_seed
            simulations[seed][str(alt_seed)]={i:{node.name: 0 for node in ND.nodes} for i in range(iterations)} #to calculate a baseline
            #diffusion=BFS_threshold(DCM,alt_seed,'pulse',time_limit=time_limit,pinned=sunits) #pin seed for steady-state modules
            #ALTERNATE based on run_network_dynamics
            diffusion=run_network_dynamics(ND,alt_seed,dsunit_map,pinned_vars=[s[:-2] for s in sunits],time_limit=time_limit-2,
                                           update=update,break_early=False,order=order,regenerate=regenerate)
            nodes_reached=set([])
            #print diffusion
            for index in range(iterations):
                if index < len(diffusion): #otherwise, nodes_reached is already at maximum
                    nodes_reached={node for node in diffusion[index] if is_sunit(node)}
                    #nodes_reached={node for node in diffusion[index] if node in DCM and DCM.node[node]['type']=='variable'}
                for node in nodes_reached:
                    if node[-1]=='1': #only record activation probability (as inactivation is the converse of this)
                        act_prob_sim[seed][index][node[:-2]]+=1
                        simulations[seed][str(alt_seed)][index][node[:-2]]=1
            #print len(nodes_reached),act_prob_sim[seed][0]
            #print act_prob_sim[seed][index]
        #find average (our label)
        act_prob_sim[seed]={index:{node:act_prob_sim[seed][index][node]/runs for node in act_prob_sim[seed][index]} for index in range(iterations)}
        #print seed,sunits,act_prob_sim[seed][iterations-1]
    
    return simulations,act_prob_sim


#compare mf probabilities against simulations to check the error in the approximation per iteration
#ASSUMES act_prob stored in dmodules (not just constants); NOTE: this also creates the simulations
#also compare only those transitions that are guaranteed to occur (p=1 or p=0) against our pathway modules
def compare_simulations(ND,seeds,dsunit_map,dmodules,dtranslator,length=1,runs=100,tau=0.0,iterations=10,
                        unknown_prob=0.5,stats={},results=True,time_limit=1,update='synchronous',order=None,regenerate=False):
    """ compares a mean-field approximation in dmodules to various simulations (runs) of the seeds 
    prints out metrics and returns the generated simulations 
    iterations refers to the number of iterations past the initial configuration at t=0 (iterations+1 total)
    NOTE: this applies only to Boolean networks with calculable DCMs and assumes steady-state perturbation """
    
    act_prob_sim={}
    simulations={} #store the random simulations per seed
    iterations=iterations+1 #include 0
    #DCM=ND.dynamics_canalization_map(output=None, )
    #p=0.5 #expected probability of unknown nodes
    tps,fps=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    tns,fns=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    accuracy,recall,precision=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    errors,avg_sim=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)] #average similarity based on jaccard index
    total=0
    nodes={node.name for node in ND.nodes}
    for seed in seeds: 
        sunits=to_list(str(seed),dsunit_map,dtranslator)
        if len(sunits)!=length:
            continue
        if len({sunit[:-2] for sunit in sunits})!=len(sunits): #there is a contradiction
            continue
        total+=1
        act_prob_sim[seed] = {i:{node: 0.0 for node in nodes} for i in range(iterations)}
        simulations[seed]={}

        #get the ground truth for the seed based on several runs
        for i in range(runs):
            #initiate a random condition
            alt_seed=[sunit for sunit in sunits]
            for node in nodes-{sunit[:-2] for sunit in alt_seed}:
                if random.random()<unknown_prob:
                    alt_seed.append(node+'-1')
                else:
                    alt_seed.append(node+'-0')
            #find the information diffusion for the sample based on the BFS threshold algorithm
            #print alt_seed
            simulations[seed][str(alt_seed)]={i:{node.name: 0 for node in ND.nodes} for i in range(iterations)} #to calculate a baseline
            #diffusion=BFS_threshold(DCM,alt_seed,'pulse',time_limit=time_limit,pinned=sunits) #pin seed for steady-state modules
            #ALTERNATE based on network_dynamics (does not require DCM calculation) #time_limit=DCM time_limit - 2 = iterations
            diffusion=run_network_dynamics(ND,alt_seed,dsunit_map,pinned_vars=[s[:-2] for s in sunits],time_limit=iterations-1,
                                           update=update,break_early=False,order=order,regenerate=regenerate) #iterations+1 above
            #if len(alt_diffusion)<time_limit-1: #diffusion has stopped early due to finding a fixed point
            #    final_step=len(alt_diffusion)-1
            #    for t in range(final_step+1,time_limit-1): alt_diffusion[t]=alt_diffusion[final_step]
            #print alt_seed,len(diffusion)#,len(alt_diffusion),config
            nodes_reached=set([])
            for index in range(iterations):
                if index < len(diffusion): #otherwise, nodes_reached is already at maximum
                    nodes_reached={node for node in diffusion[index] if is_sunit(node)}
                    #nodes_reached={node for node in diffusion[index] if node in DCM and DCM.node[node]['type']=='variable'}
                for node in nodes_reached:
                    if node[-1]=='1': #only record activation probability (as inactivation is the converse of this)
                        act_prob_sim[seed][index][node[:-2]]+=1
                        simulations[seed][str(alt_seed)][index][node[:-2]]=1
            #print len(nodes_reached),act_prob_sim[seed][0]
            #print act_prob_sim[seed][index]
        #find average (our label)
        act_prob_sim[seed]={index:{node:act_prob_sim[seed][index][node]/runs for node in act_prob_sim[seed][index]} for index in range(iterations)}
        #print seed,sunits,act_prob_sim[seed][iterations-1],dmodules[seed][iterations-1]

        #compare to our prediction
        for i in range(iterations):
            error=0.0 #error based on all node probabilities
            tp,fp,tn,fn=0.0,0.0,0.0,0.0 #keep track of statistics per iteration, based on nodes active within tau
            on_constants={node for node in act_prob_sim[seed][i] if act_prob_sim[seed][i][node]==1.0} #our constants (nodes always active)
            off_constants={node for node in act_prob_sim[seed][i] if act_prob_sim[seed][i][node]==0.0} #our constants (nodes always inactive)
            #diffusion=ip.linear_threshold(N,[str(s) for s in sorted([int(sunit[:-2]) for sunit in sunits])])
            #constants={node for ls in diffusion for node in ls} #alternate, true module at runs=infinity
            predicted_on={node for node in act_prob_sim[seed][i] if dmodules[seed][i][node]>=(1-tau)}
            predicted_off={node for node in act_prob_sim[seed][i] if dmodules[seed][i][node]<=(tau)}
            #print on_constants,off_constants,predicted_on,predicted_off
            for node in act_prob_sim[seed][i]:
                label=act_prob_sim[seed][i][node]
                error+=abs(label-dmodules[seed][i][node])**2
                #stats based on actual module, NOTE: task is constant vs non-constant
                if label==1 and dmodules[seed][i][node]>=(1-tau) or label==0 and dmodules[seed][i][node]<=(tau): #correct prediction
                    tp+=1
                elif label==1 or label==0:
                    fn+=1
                elif dmodules[seed][i][node]>=(1-tau) or dmodules[seed][i][node]<=(tau): #false prediction
                    fp+=1
                else:
                    tn+=1
        
            intersection=len(on_constants.intersection(predicted_on))+len(off_constants.intersection(predicted_off))
            union=len(on_constants.union(predicted_on))+len(off_constants.union(predicted_off))
            if union:
                avg_sim[i]+=intersection/union
            else: #no predicted or actual constants
                avg_sim[i]+=1
            #errors[i]+=math.sqrt(error)/len(act_prob_sim[seed][i]) #normalize per node
            errors[i]+=error/len(act_prob_sim[seed][i]) #FIXED FORMULA
            #print len(on_constants),len(off_constants),errors[i],avg_sim[i],error
            #print tp,fn,fp,tn
            if on_constants or off_constants:
                tps[i]+=tp/(len(on_constants)+len(off_constants))
                fns[i]+=fn/(len(on_constants)+len(off_constants))
            else: #no true positives
                tps[i]+=1
            other_set=nodes-on_constants-off_constants
            if other_set:
                fps[i]+=fp/len(other_set)
                tns[i]+=tn/len(other_set)
            else: #no true negatives
                tns[i]+=1
            #summary statistics
            accuracy[i]+=(tp+tn)/(tp+tn+fp+fn)
            if (tp+fn)!=0:
                recall[i]+=(tp)/(tp+fn)
            else: #no positives in module
                recall[i]+=1
            if (tp+fp)!=0:
                precision[i]+=(tp)/(tp+fp)
            else: #no positive predictions in module
                precision[i]+=1
    
    #normalize by number of seeds and print
    stats['tps']=[round(tps[i]/total,2) for i in range(iterations)]
    stats['fns']=[round(fns[i]/total,2) for i in range(iterations)]
    stats['fps']=[round(fps[i]/total,2) for i in range(iterations)]
    stats['tns']=[round(tns[i]/total,2) for i in range(iterations)]
    stats['accuracy']=[round(accuracy[i]/total,2) for i in range(iterations)]
    stats['recall']=[round(recall[i]/total,2) for i in range(iterations)]
    stats['precision']=[round(precision[i]/total,2) for i in range(iterations)]
    stats['avg_sim']=[round(avg_sim[i]/total,2) for i in range(iterations)]
    stats['errors']=[round(errors[i]/total,3) for i in range(iterations)]
    if results:
        print_stats(stats,total)
    
    return simulations


#compare simulations against modules by averaging over different update orders
#NOTE: will run deterministic asynchronous update if deterministic=True; stochastic otherwise
#replacement determines if sampling is done with replacement or without; argument is ignored if deterministic=True
#returns dmodules and simulations keyed by model {model: dmodules} or {model: simulations}
def compare_simulations_asynchronous(N,seeds,sunit_map,length=1,runs=100,tau=0.0,iterations=10,unknown_prob=0.5,reduced=False,ds=None,
    stats={},results=True,time_limit=1,deterministic=True,replacement=False,models=10,verbose=True,total=None):
    
    total_modules,total_simulations={},{}
    if not total: total=len(seeds)  #NOTE: seeds expected to be a list of strings
    translator={str(seed):eval(seed) for seed in seeds}  #translator will be the same between models
    stats=['errors','fns','tns','recall','avg_sim','precision','tps','fps','accuracy']
    avg_stats={stat:[0.0 for i in range(iterations+1)] for stat in stats}
    for i in range(models):
        #print "Model:",i
        dstats={}
        modules={} #reset for each model
        if deterministic: #determine the order for deterministic asynchronous
            order=[[node] for node in random_node_order(N)]
            #print [[x.name for x in block] for block in order]
            regenerate=False
            update='asynchronous'
        else: #order will be generated randomly each time step
            order=None
            regenerate=True
            if replacement: update='stochastic asynchronous'
            else: update='asynchronous'
        #run the IBMFA
        for seed in seeds:
            seed=eval(seed)
            modules[str(seed)] = run_seed_mf(N,seed,sunit_map,translator,p=unknown_prob,iterations=iterations,reduced=reduced,
                                             ds=ds,pin_start=True,update=update,order=order,regenerate=regenerate,oh=False)
        #print modules[str(seed)][1]
        #run the simulations; NOTE that all seeds must be the same length 
        simulations=compare_simulations(N,seeds,sunit_map,modules,translator,length=length,runs=runs,tau=tau,iterations=iterations,
            unknown_prob=unknown_prob,stats=dstats,results=results,time_limit=time_limit,update=update,order=order,regenerate=regenerate)
        avg_stats={stat:[avg_stats[stat][i]+dstats[stat][i] for i in range(iterations+1)] for stat in stats}
        #save the results
        total_modules[i]=modules
        total_simulations[i]=simulations
    avg_stats={stat:[round(avg_stats[stat][i]/models,3) for i in range(iterations+1)] for stat in stats}
    if verbose: print_stats(avg_stats,total=total) #NOTE: total may be incorrect if it does not equal len(seeds)
    return total_modules,total_simulations


#find the baseline: compare each simulation against all other simulations per seed
#assumes that the diffusions from the simulations are stored in the simulations dictionary
def compare_baseline(ND,seeds,dsunit_map,dtranslator,simulations,length=1,tau=0.0,iterations=10,stats={},results=True,verbose=False):
    """ compares a single simulation with all simulations of the same seed, prints out metrics and returns stats dictionary 
    NOTE: this applies only to Boolean networks """
    
    act_prob_base={seed:{} for seed in seeds}
    iterations=iterations+1 #include 0
    tps,fps=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    tns,fns=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    accuracy,recall,precision=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    errors,avg_sim=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)] #average similarity based on jaccard index
    total=0
    seed_total=0
    nodes={node.name for node in ND.nodes}
    for seed in seeds: 
        sunits=to_list(seed,dsunit_map,dtranslator)
        if len(sunits)!=length:
            continue
        if len({sunit[:-2] for sunit in sunits})!=len(sunits): #there is a contradiction
            continue
        seed_total+=1
        runs=len(simulations[seed]) #these should all be the same
    
        #compare each run against all others to get the baseline
        for run in simulations[seed]:
            total+=1
        
            act_prob_base[seed][run] = {i:{node: 0.0 for node in nodes} for i in range(iterations)}
            for i in range(iterations):
                for node in nodes:
                    act_prob_base[seed][run][i][node]=sum([simulations[seed][run2][i][node] for run2 in simulations[seed] if run!=run2])/(runs-1)     
            #print act_prob_base[seed][run]
        
            #compare to our prediction
            for i in range(iterations):
                error=0.0 #error based on all node probabilities
                tp,fp,tn,fn=0.0,0.0,0.0,0.0 #keep track of statistics per iteration, based on nodes active within tau
                on_constants={node for node in act_prob_base[seed][run][i] if act_prob_base[seed][run][i][node]==1.0} #our constants (nodes always active)
                off_constants={node for node in act_prob_base[seed][run][i] if act_prob_base[seed][run][i][node]==0.0} #our constants (nodes always inactive)
                predicted_on={node for node in act_prob_base[seed][run][i] if simulations[seed][run][i][node]>=(1-tau)}
                predicted_off={node for node in act_prob_base[seed][run][i] if simulations[seed][run][i][node]<=(tau)}
            
                for node in act_prob_base[seed][run][i]:
                    label=act_prob_base[seed][run][i][node]
                    error+=abs(label-simulations[seed][run][i][node])**2
                    #stats based on actual module, NOTE: task is constant vs non-constant
                    if label==1 and simulations[seed][run][i][node]>=(1-tau) or label==0 and simulations[seed][run][i][node]<=(tau):
                        tp+=1
                    elif label==1 or label==0:
                        fn+=1
                    elif simulations[seed][run][i][node]>=(1-tau) or simulations[seed][run][i][node]<=(tau):
                        fp+=1
                    else:
                        tn+=1
        
                intersection=len(on_constants.intersection(predicted_on))+len(off_constants.intersection(predicted_off))
                union=len(on_constants.union(predicted_on))+len(off_constants.union(predicted_off))
                avg_sim[i]+=intersection/union
                #errors[i]+=math.sqrt(error)/len(act_prob_base[seed][run][i]) #normalize per node
                errors[i]+=error/len(act_prob_base[seed][run][i]) #FIXED FORMULA
                #print tp,fn,fp,tn,errors[i]
                #print seed,act_prob_base[seed][run][i]
            
                if on_constants or off_constants:
                    tps[i]+=tp/(len(on_constants)+len(off_constants))
                    fns[i]+=fn/(len(on_constants)+len(off_constants))
                else: #no true positives
                    tps[i]+=1
                other_set=nodes-on_constants-off_constants
                if other_set:
                    fps[i]+=fp/len(other_set)
                    tns[i]+=tn/len(other_set)
                else: #no true negatives
                    tns[i]+=1
                
                #summary statistics
                accuracy[i]+=(tp+tn)/(tp+tn+fp+fn)
                if (tp+fn)!=0:
                    recall[i]+=(tp)/(tp+fn)
                else: #no positives in module
                    recall[i]+=1
                if (tp+fp)!=0:
                    precision[i]+=(tp)/(tp+fp)
                else: #no positive predictions in module
                    precision[i]+=1
    
    if verbose: print(total,'total runs,',seed_total,'seeds')
    #normalize and print
    stats['tps']=[round(tps[i]/total,2) for i in range(iterations)]
    stats['fns']=[round(fns[i]/total,2) for i in range(iterations)]
    stats['fps']=[round(fps[i]/total,2) for i in range(iterations)]
    stats['tns']=[round(tns[i]/total,2) for i in range(iterations)]
    stats['accuracy']=[round(accuracy[i]/total,2) for i in range(iterations)]
    stats['recall']=[round(recall[i]/total,2) for i in range(iterations)]
    stats['precision']=[round(precision[i]/total,2) for i in range(iterations)]
    stats['avg_sim']=[round(avg_sim[i]/total,2) for i in range(iterations)]
    stats['errors']=[round(errors[i]/total,3) for i in range(iterations)]
    if results:
        print_stats(stats,seed_total)

    return stats


#determine simulation stats for a given seed
def aggregate_simulation(seed,simulations,act_prob_sim={}):
    act_prob_sim.setdefault(seed,{})
    runs=len(simulations[seed])
    for run in simulations[seed]:
        for t in simulations[seed][run]:
            act_prob_sim[seed].setdefault(t,{})
            for node in simulations[seed][run][t]:
                act_prob_sim[seed][t].setdefault(node,0.0)
                act_prob_sim[seed][t][node]+=simulations[seed][run][t][node]
                
    #aggregate
    act_prob_sim[seed]={t:{node:act_prob_sim[seed][t][node]/runs for node in act_prob_sim[seed][t]} for t in act_prob_sim[seed]}
    return act_prob_sim


#compare results already found to simulations already found
#ASSUMES act_prob stored in modules and act_prob_sim is aggregated from simulations
#also compare only those transitions that are guaranteed to occur (p=1 or p=0) against our pathway modules
#TODO: merge this with compare_simulations above!
def compare_sim_to_modules(N,seeds,modules,act_prob_sim,tau=0.0,iterations=10,stats={},results=True):
    """ compares a mean-field approximation in dmodules to various simulations (runs) of the seeds 
    prints out metrics and returns the generated simulations 
    iterations refers to the number of iterations past the initial configuration at t=0 (iterations+1 total)
    NOTE: this applies only to Boolean networks with calculable DCMs and assumes steady-state perturbation """
    
    iterations=iterations+1 #include 0
    tps,fps=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    tns,fns=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    accuracy,recall,precision=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)],[0.0 for i in range(iterations)]
    errors,avg_sim=[0.0 for i in range(iterations)],[0.0 for i in range(iterations)] #average similarity based on jaccard index
    total=len(seeds)
    nodes={str(node.name) for node in N.nodes}
    for seed in seeds: 

        #compare to our prediction
        for i in range(iterations):
            error=0.0 #error based on all node probabilities
            tp,fp,tn,fn=0.0,0.0,0.0,0.0 #keep track of statistics per iteration, based on nodes active within tau
            on_constants={node for node in act_prob_sim[seed][i] if act_prob_sim[seed][i][node]==1.0} #our constants (nodes always active)
            off_constants={node for node in act_prob_sim[seed][i] if act_prob_sim[seed][i][node]==0.0} #our constants (nodes always inactive)#
            predicted_on={str(node) for node in modules[seed][i] if modules[seed][i][node]>=(1-tau)}
            predicted_off={str(node) for node in modules[seed][i] if modules[seed][i][node]<=(tau)}
            #print on_constants,off_constants,predicted_on,predicted_off
            
            for node in modules[seed][i]:
                label=act_prob_sim[seed][i][str(node)]
                error+=abs(label-modules[seed][i][node])**2
                #stats based on actual module, NOTE: task is constant vs non-constant
                if label==1 and modules[seed][i][node]>=(1-tau) or label==0 and modules[seed][i][node]<=(tau): #correct prediction
                    tp+=1
                elif label==1 or label==0:
                    fn+=1
                elif modules[seed][i][node]>=(1-tau) or modules[seed][i][node]<=(tau): #false prediction
                    fp+=1
                else:
                    tn+=1
        
            intersection=len(on_constants.intersection(predicted_on))+len(off_constants.intersection(predicted_off))
            union=len(on_constants.union(predicted_on))+len(off_constants.union(predicted_off))
            if union:
                avg_sim[i]+=intersection/union
            else: #no predicted or actual constants
                avg_sim[i]+=1
            #errors[i]+=math.sqrt(error)/len(act_prob_sim[seed][i]) #normalize per node
            errors[i]+=error/len(act_prob_sim[seed][i]) #FIXED FORMULA
            #print len(on_constants),len(off_constants),errors[i],avg_sim[i],error
            #print tp,fn,fp,tn
            if on_constants or off_constants:
                tps[i]+=tp/(len(on_constants)+len(off_constants))
                fns[i]+=fn/(len(on_constants)+len(off_constants))
            else: #no true positives
                tps[i]+=1
            other_set=nodes-on_constants-off_constants
            if other_set:
                fps[i]+=fp/len(other_set)
                tns[i]+=tn/len(other_set)
            else: #no true negatives
                tns[i]+=1
            #summary statistics
            accuracy[i]+=(tp+tn)/(tp+tn+fp+fn)
            if (tp+fn)!=0:
                recall[i]+=(tp)/(tp+fn)
            else: #no positives in module
                recall[i]+=1
            if (tp+fp)!=0:
                precision[i]+=(tp)/(tp+fp)
            else: #no positive predictions in module
                precision[i]+=1
    
    #normalize and print
    stats['tps']=[round(tps[i]/total,2) for i in range(iterations)]
    stats['fns']=[round(fns[i]/total,2) for i in range(iterations)]
    stats['fps']=[round(fps[i]/total,2) for i in range(iterations)]
    stats['tns']=[round(tns[i]/total,2) for i in range(iterations)]
    stats['accuracy']=[round(accuracy[i]/total,2) for i in range(iterations)]
    stats['recall']=[round(recall[i]/total,2) for i in range(iterations)]
    stats['precision']=[round(precision[i]/total,2) for i in range(iterations)]
    stats['avg_sim']=[round(avg_sim[i]/total,2) for i in range(iterations)]
    stats['errors']=[round(errors[i]/total,3) for i in range(iterations)]
    if results:
        print_stats(stats,total)
    
    return stats

