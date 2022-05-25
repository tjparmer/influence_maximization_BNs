#Driver set selection
#Thomas Parmer, 2022

from utils import reduce_seeds
from mean_field_computations import *
from entropy_computations import *
from simulations import *

#returns a set of s-units by name that are resolved based on tau from a given node: act_prob mapping
#NOTE: this can return multiple variable states if tau is sufficiently large (which it should not be)
def resolved_sunits(node_probs,tau):
    
    sunit_set=set([])
    for node in node_probs:
        if node_probs[node]>=1-tau: sunit_set.add(str(node)+'-1') #inclusive
        if node_probs[node]<=0+tau: sunit_set.add(str(node)+'-0') #inclusive
    return sunit_set


#for a given set of s-units, determine if the given attractor is reachable
def attractor_is_reachable(sunit_set,attractors,key):
    return not sunit_set-attractors[key] #active nodes are a subset of the attractor


#for a given set of s-units, determines which attractors are still reachable
def reachable_attractors(sunit_set,attractors):
    
    #NOTE: this assumes active nodes are present in the attractor, and hence pinning control
    reachables={}
    for attractor in attractors:
        if attractor_is_reachable(sunit_set,attractors,attractor): #helper module to check single attractor
            reachables[attractor]=attractors[attractor]
            
    return reachables #,len(reachables)/len(attractors)


#determine seed selection to best reduce entropy in the system
#strategies: random, greedy, top x; may solve for specific attractor
#WARNING: if passing modules, make sure the update scheme used is consistent with the parameters passed to this function
def top_selection(N,sunits,sunit_map,modules={},translator={},seeds=None,max_s=10,top=1,reduced=False,ds=None,pinning={},tau=0.0,
        iterations=10,unknown_prob=0.5,t=10,base=2,normalized=True,attractor=None,attractors=None,pin_start=True,
        force=False,update='synchronous',order=None,regenerate=False,models=1,drivers=False,names='string',start_seed=()):
    """ determines seed selection across seed sizes to most reduce entropy in the system, 
    considers the top x per seed size s; returns selections and the entropies dictionary 
    if attractor is given (requires attractors dictionary), only considers seeds that may resolve in that attractor 
    t is the iteration to compare entropies on
    seeds specifies which nodes to use to try to reduce entropy in the network (e.g. all or fvs only)
    drivers = True allows for early exit of the function to return the selection once it reaches zero entropy
    set names = 'string' for biological networks, 'int' for RBNs
    start_seed allows the selection to start with a given seed (e.g. input nodes), pass as a tuple """

    if not seeds:
        if not modules:
            modules,translator=find_modules(N,1,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
                        iterations=iterations,data=True,seeds=None,p=unknown_prob,verbose=False,pin_start=pin_start,
                        update=update,order=order,regenerate=regenerate,names=names)
        seeds=reduce_seeds(modules,sunit_map,translator,length=1) #all s-units in the network
    start_length=len(to_list(start_seed,sunit_map,translator))
    candidates,selections=set([str(start_seed)]),[str(start_seed)] #top level candidates to reduce entropy; starts with no selection by default
    diffusion=find_modules(N,start_length,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,iterations=iterations,
        data=True,seeds=[start_seed],p=unknown_prob,verbose=False,pin_start=pin_start,update=update,order=order,regenerate=regenerate,names=names)[0][str(start_seed)]
    seed_entropy,seed_configs=config_entropy(diffusion,base=base,normalized=normalized)
    entropies={str(start_seed): seed_entropy[t]}
    if drivers and entropies[str(start_seed)]==0: #early exit if start seed is a driver set
        return selections,entropies
    #print len(modules),len(translator)

    for s in range(start_length+1,max_s+1):
        for module in candidates:
            for single in seeds:
                neg,pos=int(translator[single][0]/2)*2,int(translator[single][0]/2)*2+1 #OFF and ON state; assumes binary!
                if neg in translator[module] or pos in translator[module]: continue #repetition or contradiction
                seed=tuple(sorted(set(translator[module]).union(set(translator[single]))))
                if not str(seed) in modules:
                    modules,translator=find_modules(N,s,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
                            iterations=iterations,data=True,seeds=[seed],p=unknown_prob,verbose=False,pin_start=pin_start,
                            update=update,order=order,regenerate=regenerate,names=names)
                if attractor: #skip all seeds that won't resolve in the given attractor
                    sunit_set=resolved_sunits(modules[str(seed)][t],tau=tau)
                    if type(attractor)==str: #represents a fixed point
                        if not attractor_is_reachable(sunit_set,attractors,attractor): continue
                    else:
                        lc_configs={a:attractors[attractor][i] for i,a in enumerate(attractor)}
                        #print [lc_configs[a] for a in attractor],sum([attractor_is_reachable(sunit_set,lc_configs,a) for a in attractor])
                        if sum([attractor_is_reachable(sunit_set,lc_configs,a) for a in attractor])<len(attractor):
                            continue #checks every config in limit cycle (s-units must be fixed in all attractor configs)
                seed_entropy,seed_configs=config_entropy(modules[str(seed)],base=base,normalized=normalized)
                entropies[str(seed)]=seed_entropy[t] #select based on final entropy if t=iterations
        if force: #enforce selection every iteration; will be random if new node does not reduce entropy
            candidates=reduce_seeds(entropies,sunit_map,translator,length=s) #force a new selection every iteration
            candidates=set(sorted(candidates,key=lambda x: entropies[x])[:top])
        else: 
            candidates=set(sorted(entropies,key=lambda x: entropies[x])[:top])
        if candidates:
            min_candidate=min(candidates,key=lambda x: entropies[x])
            selections.append(min_candidate) #select the one with lowest entropy
            if drivers and min_candidate in entropies and entropies[min_candidate]==0:
                return selections,entropies #early exit in looking for driver sets
        #print [to_list(seed,sunit_map,translator) for seed in sorted(entropies,key=lambda x: entropies[x])[:top]]
        #for seed in candidates: print s,candidates,selections,entropies[seed] #,sorted(entropies,key=lambda x: entropies[x]),
    return selections,entropies


#reduce given seed to a set of driver nodes; goal is to find minimal sets that reach an attractor or target entropy
def reduce_selection(seed,N,sunits,sunit_map,modules={},translator={},reduced=False,ds=None,pinning={},
                     iterations=10,unknown_prob=0.5,t=10,base=2,normalized=True,pin_start=True,
                     update='synchronous',order=None,regenerate=False,names='string'):
    """ reduce seed in modules to a subset by removing any nodes that do not increase entropy when removed,
    checks entropy at the specified time step t, returns the reduced set as a seed (tuple)
    NOTE: seed must be passed as a list if it's not in translator """
    
    if not str(seed) in modules or not modules[str(seed)]: #check for missing or empty modules
        #NOTE: s arbitrarily set to 1 in find_modules because its value doesn't matter
        modules,translator=find_modules(N,1,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
                        iterations=iterations,data=True,seeds=[seed],p=unknown_prob,verbose=False,pin_start=pin_start,
                            update=update,order=order,regenerate=regenerate,names=names)
    seed_entropy,seed_configs=config_entropy(modules[str(seed)],base=base,normalized=normalized)
    min_entropy=seed_entropy[t] #select based on final entropy if t=iterations
    seed=translator[str(seed)] #force seed to be in list form
    
    #test removing nodes one by one
    drivers=set([])
    removed=set([]) #alternatively, remove sunits sequentially from the seed
    for sunit in seed:
        alt_seed=tuple(sorted(set(seed)-{sunit}-removed)) #updated to remove sequentially
        #print seed,sunit,alt_seed
        if not str(alt_seed) in modules or not modules[str(alt_seed)]: #check for missing or empty modules
            modules,translator=find_modules(N,1,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
                    iterations=iterations,data=True,seeds=[alt_seed],p=unknown_prob,verbose=False,pin_start=pin_start,
                    update=update,order=order,regenerate=regenerate,names=names)
        seed_entropy,seed_configs=config_entropy(modules[str(alt_seed)],base=base,normalized=normalized)
        if seed_entropy[t]>min_entropy: #this is a driver node because it reduces entropy
            drivers.add(sunit)
        else: #alternative
            removed.add(sunit)
    return tuple(sorted(drivers)) #note that this should already be sorted
    

#determine seed selection to best reduce entropy in the system
#strategies: random, greedy, top x; may solve for specific attractor
#TODO: make compatible with different updating schemes
def top_selection_simulations(N,sunits,sunit_map,modules={},translator={},seeds=None,max_s=10,top=1,reduced=False,ds=None,pinning={},tau=0.0,
                  iterations=10,unknown_prob=0.5,t=10,base=2,normalized=True,attractor=None,attractors=None,pin_start=True,force=False,runs=100,
                             simulations={},act_prob_sim={},names='string'):
    """ determines seed selection across seed sizes to most reduce entropy in the system based on simulations, 
    considers the top x per seed size s; returns selections and the entropies dictionary 
    if attractor is given (requires attractors dictionary), only considers seeds that may resolve in that attractor 
    t is the iteration to compare entropies on """
    
    tl=iterations+2
    #simulations,act_prob_sim={},{}
    if not seeds:
        if not modules:
            modules,translator=find_modules(N,1,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
                    iterations=iterations,data=True,seeds=None,p=unknown_prob,verbose=False,names=names)
        seeds=reduce_seeds(modules,sunit_map,translator,length=1)
        simulations,act_prob_sim=create_simulations(N,seeds,sunit_map,translator,length=1,runs=runs,iterations=iterations,
                                           unknown_prob=unknown_prob,time_limit=tl)
        seeds=reduce_seeds(modules,sunit_map,translator,length=1) #all s-units in the network
    candidates,selections=set(['()']),['()'] #top level candidates to reduce entropy; starts with no selection by default
    simulations,act_prob_sim=create_simulations(N,['()'],sunit_map,translator,length=0,runs=runs,iterations=iterations,
                                           unknown_prob=unknown_prob,time_limit=tl,simulations=simulations,act_prob_sim=act_prob_sim)
    #diffusion=find_modules(N,0,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,iterations=iterations,
    #                       data=True,seeds=[()],p=unknown_prob,verbose=False)[0]['()']
    seed_entropy,seed_configs=config_entropy(act_prob_sim['()'],base=base,normalized=normalized)
    entropies={'()': seed_entropy[t]}
    print(len(modules),len(translator))
    #print simulations.keys(),act_prob_sim.keys()

    for s in range(1,max_s+1):
        for module in candidates:
            for single in seeds:
                neg,pos=int(translator[single][0]/2)*2,int(translator[single][0]/2)*2+1 #OFF and ON state; assumes binary!
                if neg in translator[module] or pos in translator[module]: continue #repetition or contradiction
                seed=tuple(sorted(set(translator[module]).union(set(translator[single]))))
                if not str(seed) in act_prob_sim:
                    simulations,act_prob_sim=create_simulations(N,[str(seed)],sunit_map,translator,length=s,runs=runs,iterations=iterations,
                                           unknown_prob=unknown_prob,time_limit=tl,simulations=simulations,act_prob_sim=act_prob_sim)
                    #modules,translator=find_modules(N,s,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
                    #        iterations=iterations,data=True,seeds=[seed],p=unknown_prob,verbose=False,pin_start=pin_start)
                if attractor: #skip all seeds that won't resolve in the given attractor
                    sunit_set=resolved_sunits(act_prob_sim[str(seed)][t],tau=tau)
                    if not attractor_is_reachable(sunit_set,attractors,attractor): continue
                seed_entropy,seed_configs=config_entropy(act_prob_sim[str(seed)],base=base,normalized=normalized)
                entropies[str(seed)]=seed_entropy[t] #select based on final entropy if t=iterations
        if force: #enforce selection every iteration; will be random if new node does not reduce entropy
            candidates=reduce_seeds(entropies,sunit_map,translator,length=s) #force a new selection every iteration
            candidates=set(sorted(candidates,key=lambda x: entropies[x])[:top])
        else: 
            candidates=set(sorted(entropies,key=lambda x: entropies[x])[:top])
        if candidates:
            selections.append(min(candidates,key=lambda x: entropies[x])) #select the one with lowest entropy
        #print [to_list(seed,sunit_map,translator) for seed in sorted(entropies,key=lambda x: entropies[x])[:top]]
        #for seed in candidates: print s,candidates,selections,entropies[seed] #,sorted(entropies,key=lambda x: entropies[x]),
    return selections,entropies


#reduce given seed to a set of driver nodes; goal is to find minimal sets that reach an attractor or target entropy
#TODO: make compatible with different updating schemes
def reduce_selection_simulations(seed,N,sunits,sunit_map,translator={},time_limit=12,iterations=10,unknown_prob=0.5,
                                 t=10,base=2,normalized=True,simulations={},act_prob_sim={},length=1,runs=100):
    """ reduce seed in modules to a subset by removing any nodes that do not increase entropy when removed based on simulations,
    checks entropy at the specified time step t, returns the reduced set as a seed (tuple)
    NOTE: seed must be passed as a list if it's not in translator """
    
    if not str(seed) in act_prob_sim:
        #NOTE: s arbitrarily set to 1 in find_modules because its value doesn't matter
        #modules,translator=find_modules(N,1,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
        #                                iterations=iterations,data=True,seeds=[seed],p=unknown_prob,verbose=False)
        simulations,act_prob_sim=create_simulations(N,[str(seed)],sunit_map,translator,length=length,runs=runs,iterations=iterations,
                                unknown_prob=unknown_prob,time_limit=time_limit,simulations=simulations,act_prob_sim=act_prob_sim)
    seed_entropy,seed_configs=config_entropy(act_prob_sim[str(seed)],base=base,normalized=normalized)
    min_entropy=seed_entropy[t] #select based on final entropy if t=iterations
    seed=translator[str(seed)] #force seed to be in list form
    
    #test removing nodes one by one
    drivers=set([])
    for sunit in seed:
        alt_seed=tuple(sorted(set(seed)-{sunit}))
        if not str(alt_seed) in act_prob_sim:
            simulations,act_prob_sim=create_simulations(N,[str(alt_seed)],sunit_map,translator,length=length-1,runs=runs,iterations=iterations,
                                unknown_prob=unknown_prob,time_limit=time_limit,simulations=simulations,act_prob_sim=act_prob_sim)
            #modules,translator=find_modules(N,1,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
            #                            iterations=iterations,data=True,seeds=[alt_seed],p=unknown_prob,verbose=False)
        seed_entropy,seed_configs=config_entropy(act_prob_sim[str(alt_seed)],base=base,normalized=normalized)
        if seed_entropy[t]>min_entropy: #this is a driver node because it reduces entropy
            drivers.add(sunit)
    return tuple(sorted(drivers)) #note that this should already be sorted
    

#driver set for feedback vertex set theory
#NOTE: we assume this takes a BooleanNetwork and so the node indices will match
def fvs_set(N,method=None):
    if method:
        fvs=N.feedback_vertex_set_driver_nodes(method=method) #method='bruteforce'
    else: #use default
        fvs=N.feedback_vertex_set_driver_nodes()
    fset=set([])
    for x in fvs:
        for node in sorted(x):
            #print node,ND.nodes[node].name,dsunit_map[node*2],dsunit_map[node*2+1]
            fset.add('('+str(node*2)+',)') #negative state
            fset.add('('+str(node*2+1)+',)') #positive state
    return fset


#random seed selection strategy; single sample only
def random_selection(N,sunits,sunit_map,modules={},translator={},seeds=None,max_s=10,entropies={},reduced=False,ds=None,
            pinning={},tau=0.0,iterations=10,unknown_prob=0.5,t=10,base=2,normalized=True,attractor=None,attractors=None,
                    update='synchronous',order=None,regenerate=False,names='string'):
    """ determines seed selection across seed sizes randomly by adding a new s-unit at each iteration, 
    returns selections and the entropies dictionary (which may also be passed in as an argument)
    if attractor is given, only considers seeds that may resolve in that attractor """
    
    if not seeds:
        if not modules:
            modules,translator=find_modules(N,1,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
                iterations=iterations,data=True,seeds=None,p=unknown_prob,verbose=False,update=update,order=order,regenerate=regenerate,names=names)
        seeds=reduce_seeds(modules,sunit_map,translator,length=1) #all s-units in the network
    selections_rnd,chosen=['()'],set([])
    for s in range(1,max_s+1):
        choice=random.choice(list(seeds-chosen)) #sample without replacement
        if s==0: new_set=choice #add to selections_rnd
        if s>0: #add on to previously chosen set 
            new_set=tuple(sorted([x for x in {translator[choice][0]}.union(set(translator[selections_rnd[-1]]))]))
        selections_rnd.append(str(new_set))
        neg,pos='('+str(int(translator[choice][0]/2)*2)+',)','('+str(int(translator[choice][0]/2)*2+1)+',)' #assumes binary!
        chosen.add(neg) #remove the OFF version of the selected s-unit
        chosen.add(pos) #remove the ON version of the selected s-unit
        if str(new_set) not in modules:
            modules,translator=find_modules(N,s,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,pinning=pinning,
                iterations=iterations,data=True,seeds=[new_set],p=unknown_prob,verbose=False,update=update,order=order,regenerate=regenerate,names=names)
        if attractor: #skip all seeds that won't resolve in the given attractor
            sunit_set=resolved_sunits(modules[str(seed)][t],tau=tau)
            if not attractor_is_reachable(sunit_set,attractors,attractor): continue
        seed_entropy,seed_configs=config_entropy(modules[str(new_set)],base=base,normalized=normalized)
        entropies[str(new_set)]=seed_entropy[t] #select based on final entropy if t=iterations
    return selections_rnd,entropies


### find entropy per s given strategy
def strategy_entropy(selections,N,sunit_map,modules,translator,runs=100,tau=0.0,iterations=10,unknown_prob=0.5,stats={},
        results=False,time_limit=12,act_prob_sim={},entropies_sim={},t=10,base=2,normalized=True,update='synchronous',order=None,regenerate=False):
    """ determines the entropy of the seed selections across seed sizes, returns entropies_sim dictionary """

    for i,seed in enumerate(selections):
        if not seed in act_prob_sim:
            length=len(to_list(seed,sunit_map,translator))
            simulation=compare_simulations(N,[seed],sunit_map,modules,translator,length=length,runs=runs,tau=tau,iterations=iterations,
                unknown_prob=unknown_prob,stats=stats,results=results,time_limit=time_limit,update=update,order=order,regenerate=regenerate)
            aggregate_simulation(seed,simulation,act_prob_sim)
        seed_entropy,seed_configs=config_entropy(act_prob_sim[str(seed)],base=base,normalized=normalized)
        entropies_sim[str(seed)]=seed_entropy[t] #select based on final entropy
    return entropies_sim


#find average of many random strategies
def avg_random_selections(N,sunits,sunit_map,modules={},translator={},samples=1,seeds=None,max_s=10,entropies={},reduced=False,
        ds=None,pinning={},tau=0.0,iterations=10,unknown_prob=0.5,t=10,base=2,normalized=True,attractor=None,runs=100,stats={},
        results=False,time_limit=12,act_prob_sim={},entropies_sim={},attractors=None,update='synchronous',order=None,regenerate=False):
    
    avg,selection_ls={i:0.0 for i in range(max_s+1)},[]
    for i in range(samples):
        selections,entropies=random_selection(N,sunits,sunit_map,modules=modules,translator=translator,seeds=seeds,max_s=max_s,
            entropies=entropies,reduced=reduced,ds=ds,pinning=pinning,tau=tau,iterations=iterations,attractors=attractors,
            unknown_prob=unknown_prob,t=t,base=base,normalized=normalized,attractor=attractor,update=update,order=order,regenerate=regenerate)
        selection_ls.append(selections) #record all random selections
        #test against simulations
        entropies_sim=strategy_entropy(selections,N,sunit_map,modules,translator,runs=runs,tau=tau,iterations=iterations,
            unknown_prob=unknown_prob,stats=stats,results=results,time_limit=time_limit,act_prob_sim=act_prob_sim,
            entropies_sim=entropies_sim,t=t,base=base,normalized=normalized,update=update,order=order,regenerate=regenerate)
        #print selections,entropies[selections[-1]],entropies_sim[selections[-1]] #final entropy
        for i in avg:
            avg[i]+=entropies_sim[selections[i]]
    avg={i:avg[i]/samples for i in avg}
        
    return selection_ls,avg


#returns the driver set based on the top selection greedy heuristic 
#NOTE: returns None if no driver set found, () if empty set found
def driver_selection(N,sunits,sunit_map,modules={},translator={},seeds=None,max_s=10,top=1,reduced=False,ds=None,pinning={},tau=0.0,
    iterations=10,unknown_prob=0.5,t=10,base=2,normalized=True,attractor=None,attractors=None,pin_start=True,force=False,
        update='synchronous',order=None,regenerate=False,names='string',start_seed=()):
    """ return the driver set if possible based on the top selection greedy heuristic"""

    selections,entropies=top_selection(N,sunits,sunit_map,modules,translator,seeds=seeds,max_s=max_s,top=top,reduced=reduced,ds=ds,
        pinning=pinning,tau=tau,iterations=iterations,unknown_prob=unknown_prob,t=t,base=base,normalized=normalized,attractor=attractor,
        attractors=attractors,pin_start=pin_start,force=force,update=update,order=order,regenerate=regenerate,drivers=True,names=names,start_seed=start_seed)
    #print selections,entropies
    for seed in selections: 
        if entropies[seed]==0.0:
            rs=reduce_selection(seed,N,sunits,sunit_map,modules,translator,reduced=reduced,ds=ds,t=t,
                iterations=iterations,pin_start=pin_start,update=update,order=order,regenerate=regenerate,names=names)
            return rs

