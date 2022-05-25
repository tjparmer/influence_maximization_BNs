#Mean-field approximations
#Thomas Parmer, 2022

from itertools import chain, combinations
import pandas as pd
from modules import look_up_table
import random
from utils import *


#LUT pre-processing to only consider the rows that output to 1 or 0 (whichever state is in the minority)
#NOTE: this only works for Boolean outputs, and it chooses 1 if there is a tie
def LUT_reduce(df):
    """ reduces a LUT (given as a dataframe) to only rows with the minority output """
    
    state=df['Out:'].value_counts().idxmin()
    if len(df['Out:'].value_counts())==1: #we have a constant output so no minimum rows
        state=1-state #df['Out:'].value_counts().argmin()
    return pd.DataFrame([row[1:] for row in df.itertuples() if row[2]==state]) #removes first index


#compare general networks using a mean-field approximation
def mf_approx(node,act_prob,i,state=1,reduced=False,ds=None):
    """ mean-field approximation for the activation probability of the given node to the given state at iteration i,
    based on the probabilities of node states in its input configurations 
    
    The reduced argument indicates that LUTs have been reduced to their minority output (only works for binary outputs),
    which is stored in the ds data structure
    
    This function returns the new probability and the number of config rows that result in the given (or reduced) state """
    
    s=0.0
    t=0 #number of rows we check
    if not node.inputs: #constant with no inputs, so no way to update its state
        return act_prob[i-1][node.name],0
    if reduced and not len(look_up_table(node,ds=ds)): #this node has constant output
        if state==node.outputs[0]: return 1.0,0
        else: return 0.0,0
        
    for row in look_up_table(node,ds=ds).itertuples(): #look at each input configuration
        #solve equation based on probabilities of inputs
        #NOTE: with reduced, we look at all rows because the LUT has already been reduced
        if row[2]==state or reduced: #ignore those that don't lead to the required state activation
            config=row[1]
            p=1.0
            t+=1
            #TODO: fix this to work with non-binary inputs
            for j,inp in enumerate(config):
                if int(inp)==0: #contribution from OFF node
                    p*=1-act_prob[i-1][node.inputs[j]]
                else: #contribution from ON node
                    p*=act_prob[i-1][node.inputs[j]]
            s+=p
            
    if reduced and row[2]!=state: #only have to check one row because all row outputs are the same
        return 1-s,t
    
    return s,t


#synchronous update
#pinning allows to pin a certain state (e.g. 0,1), pin_start allows to pin the seed
def synchronous_mf(N,act_nodes,act_prob,iterations=10,pinning=set([]),reduced=False,ds=None,pin_start=True):
    
    #iterate through the mean-field approximation
    for i in range(1,iterations+1):
        for node in N.nodes:
            if node.name in act_nodes and pin_start: #pin the starting nodes
                act_prob[i][node.name]=act_prob[i-1][node.name]
                continue
            s,t=mf_approx(node,act_prob,i,reduced=reduced,ds=ds) #solve equation based on probabilities of inputs
            if s>1: s=1.0 #fix rounding errors
            if s<0: s=0.0 #fix rounding errors
            #allow pinning when state is reached
            if (0 in pinning) and act_prob[i-1][node.name]==0: act_prob[i][node.name]=0.0
            elif (1 in pinning) and act_prob[i-1][node.name]==1: act_prob[i][node.name]=1.0
            else:
                act_prob[i][node.name]=s
            if i==iterations: 
                pass
                #print node.name,act_prob[i][node.name],t,2**node.k,node.inputs
    return act_prob


#generate a random node order
def random_node_order(N):
    nodes=[n for n in N.nodes] #so that the original list isn't changed
    random.shuffle(nodes)
    return nodes


#create an order variable from a list of node names
def generate_blocks(N,order):
    new_order=[]
    for block in order:
        new_block=[node for name in block for node in N.nodes if node.name==name]
        new_order.append(new_block)
    return new_order


#deterministic asynchronous update and stochastic with the constraint that every node is updated between iterations
#pinning allows to pin a certain state (e.g. 0,1), pin_start allows to pin the seed
#order dictates the order to update nodes, allows for sequential block updating
#defaut is random order; set regenerate=True to pick a new random order for each iteration
def asynchronous_mf(N,act_nodes,act_prob,iterations=10,pinning=set([]),reduced=False,ds=None,pin_start=True,order=None,
                    regenerate=False,oh=False):

    if not order:
        order=[[node] for node in random_node_order(N)] #random by default
    order_history={} #track order to replicate
    #iterate through the mean-field approximation
    current_state={0:{node.name:act_prob[0][node.name] for node in N.nodes}} #state based on which to make updates
    for i in range(1,iterations+1):
        order_history[i]=order
        #print [[node.name for node in block] for block in order]
        #print "Iteration:",i,current_state
        for block in order:
            for node in block:
                if node.name in act_nodes and pin_start: #pin the starting nodes
                    act_prob[i][node.name]=act_prob[i-1][node.name]
                    continue
                #NOTE: we hard-code 1 below to pass current_state as if it were act_prob
                s,t=mf_approx(node,current_state,1,reduced=reduced,ds=ds) #solve equation based on probabilities of inputs
                if s>1: s=1.0 #fix rounding errors
                if s<0: s=0.0 #fix rounding errors
                #allow pinning when state is reached
                if (0 in pinning) and act_prob[i-1][node.name]==0: act_prob[i][node.name]=0.0
                elif (1 in pinning) and act_prob[i-1][node.name]==1: act_prob[i][node.name]=1.0
                else:
                    act_prob[i][node.name]=s
                if i==iterations: 
                    pass
                    #print node.name,act_prob[i][node.name],t,2**node.k,node.inputs
            #update the current state once the block is completed
            for node in block: current_state[0][node.name]=act_prob[i][node.name]
            #print current_state,node.name,s
        if regenerate:
            order=[[node] for node in random_node_order(N)] #new random order
            
    if oh: return act_prob,order_history
    return act_prob


#stochastic asynchronous update
#pinning allows to pin a certain state (e.g. 0,1), pin_start allows to pin the seed
#order is chosen randomly one node at a time
def stochastic_asynchronous_mf(N,act_nodes,act_prob,iterations=10,pinning=set([]),reduced=False,ds=None,pin_start=True,oh=False):
 
    order=[random.choice(N.nodes) for node in range(len(N.nodes))] #so that iterations will be as long as with other updating schemes
    order_history={} #track order to replicate
    #iterate through the mean-field approximation
    for i in range(1,iterations+1):
        order_history[i]=order #track order to replicate
        act_prob[i]={node.name:act_prob[i-1][node.name] for node in N.nodes} #state based on which to make updates
        #print [node.name for node in order]
        #print "Iteration:",i,act_prob[i]
        for node in order:
            if node.name in act_nodes and pin_start: #pin the starting nodes
                act_prob[i][node.name]=act_prob[i-1][node.name]
                continue
            #NOTE: we pass i+1 below to pass the current state of act_prob
            s,t=mf_approx(node,act_prob,i+1,reduced=reduced,ds=ds) #solve equation based on probabilities of inputs
            if s>1: s=1.0 #fix rounding errors
            if s<0: s=0.0 #fix rounding errors
            #allow pinning when state is reached
            if (0 in pinning) and act_prob[i][node.name]==0: act_prob[i][node.name]=0.0
            elif (1 in pinning) and act_prob[i][node.name]==1: act_prob[i][node.name]=1.0
            else:
                act_prob[i][node.name]=s
            if i==iterations: 
                pass
                #print node.name,act_prob[i][node.name],t,2**node.k,node.inputs
            #print act_prob[i],node.name,s
        order=[random.choice(N.nodes) for node in range(len(N.nodes))] #new random order
        #order_history[i]=order
    if oh: return act_prob,order_history
    return act_prob


#find top modules by removing those whose unfolding is a subset of another
#NOTE: this only checks the resulting set and therefore only works with pinning perturbation
def top_modules(modules,translator,s=1):
    top_modules={}
    #iterate through modules, largest first
    for seed in sorted(modules,key=lambda x: len(modules[x]),reverse=True):
        if len(translator[seed])!=s: continue #only subsume for a given s value
        subsumed=False 
        for tm in top_modules:
            if not modules[seed] - modules[tm]: #the seed is a subset of a top module already seen
                subsumed=True
                break
        
        if not subsumed:
            top_modules[seed] = modules[seed]
        
    return top_modules


#solution from https://stackoverflow.com/questions/374626/how-can-i-find-all-the-subsets-of-a-set-with-exactly-n-elements
def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))


#find synergistsic modules by removing those whose seed combination does not add synergy
#NOTE: assumes that submodules are already included in modules
#NOTE: based on powersets, requires a standard numbering convention to check the lengths of submodules
def syn_modules(tm,modules,translator,s=1):
    """ checks for synergy within tm (top modules) by checking submodules in modules using 
    translator to map module strings to lists, given seed size s """
    
    sm={}
    for module in tm:
        seed = translator[module]
        if len(seed)!=s: continue
        #seeds of length 1 are automatically synergistic
        if s==1 and len(seed)==1:
            sm[str(seed)] = modules[str(seed)]
            continue
            
        #find powerset for seeds of length > 1
        ps=list(powerset(seed))
        synergistic=True
        for i in range(1,int(len(ps)/2)):
            set1=str(ps[i])
            set2=str(ps[len(ps)-i-1])
            joint_set=modules[set1].union(modules[set2])
            #print i,seed,set1,set2,modules[str(seed)],joint_set
            if not(modules[str(seed)] - joint_set):
                synergistic=False
                break
        if synergistic:
            sm[str(seed)] = modules[str(seed)]
        
    return sm


#find modules that are both maximal and synergistic given a seed size s
def info_modules(modules,translator,s=1):
    
    tm = top_modules(modules,translator,s)
    return syn_modules(tm,modules,translator,s)


#initialize an activation probability table for a given seed, given a starting probability p and number of iterations
def mf_seed(N,seed,sunit_map,translator,p=0.5,iterations=10,names='string'):
    
    if not str(seed) in translator:
        translator[str(seed)] = seed
    act_units=to_list(seed,sunit_map,translator) #list of activated s-unit names
    if names=='string':
        act_nodes={sunit[:-2] for sunit in act_units}
    else:
        act_nodes={int(sunit[:-2]) for sunit in act_units}
    act_prob = {i: {node.name: p for node in N.nodes} for i in range(iterations+1)}
    #print seed, act_units
    for sunit in act_units:
        if names=='string':
            var=sunit[:-2]
        else:
            var=int(sunit[:-2])
        if sunit[-1]=='0': #MODIFIED, this was causing an issue with variable names like 'cdk-0'
            act_prob[0][var]=0.0
        elif sunit[-1]=='1':
            act_prob[0][var]=1.0
    return act_nodes,act_prob


#run the IBMFA for a seed, given an update mode
def run_seed_mf(N,seed,sunit_map,translator,p=0.5,iterations=10,pinning=set([]),reduced=False,ds=None,pin_start=True,
            update='synchronous',order=None,regenerate=False,oh=False,names='string'):
    
    act_nodes,act_prob=mf_seed(N,seed,sunit_map,translator,p=p,iterations=iterations,names=names)
    if update=='synchronous':
        return synchronous_mf(N,act_nodes,act_prob,iterations,pinning,reduced,ds,pin_start)
    if update=='asynchronous':
        return asynchronous_mf(N,act_nodes,act_prob,iterations,pinning,reduced,ds,pin_start,order,regenerate,oh)
    if update=='stochastic asynchronous':
        return stochastic_asynchronous_mf(N,act_nodes,act_prob,iterations,pinning,reduced,ds,pin_start,oh)


#run the IBMFA for a seed, averaging over possible update schedules
#if models=1, this replicates run_seed_mf
def average_seed_mf(N,seed,sunit_map,translator,p=0.5,iterations=10,pinning=set([]),reduced=False,ds=None,pin_start=True,
            update='synchronous',order=None,regenerate=False,oh=False,models=1,names='string'):
    
    avg_prob = {i: {node.name: 0.0 for node in N.nodes} for i in range(iterations+1)} #average act_prob
    for m in range(models):
        act_prob=run_seed_mf(N,seed,sunit_map,translator,p,iterations,pinning,reduced,ds,pin_start,update,order,regenerate,oh,names=names)
        avg_prob={i: {node.name: avg_prob[i][node.name]+act_prob[i][node.name] for node in N.nodes} for i in range(iterations+1)}
    return {i: {node.name: avg_prob[i][node.name]/models for node in N.nodes} for i in range(iterations+1)}


#find all pathway modules per given seed size s using the mean-field approximation
def find_modules(N,s=1,sunits=None,sunit_map=None,modules={},translator={},iterations=10,tau=0.00001,pinning=set([]),
                 reduced=False,ds=None,p=0.5,seeds=None,data=False,samples=None,verbose=True,pin_start=True,
                update='synchronous',order=None,regenerate=False,models=1,names='string'):
    """ find all pathway modules for a given network N and seed size s, with the given modules and translator,
    can iteratively add to modules for different s values; based on the parameters iterations and tau
    
    set pinning to True to make the assumption that once states are resolved, they never change,
    set reduced to true if the LUTs of the network components have been reduced,
    set ds to a data structure holding the LUTs of the network components to increase speed of computation
    set p to the default probability for each node that is not in the seed 
    set seeds to a list of which seeds you want to find modules for (by s-unit number)
    set data to True to return act_prob in place of module sets 
    set pin_start to True to pin the starting nodes every iteration
    ASSUMES node names are strings; set names='int' if this is not the case """
    
    #define s-units if they are not defined outside the function
    if not sunits or not sunit_map:
        sunits,sunit_map = get_sunits(N)
    #sunits={i for i in range(len(lsunits))}
    #define seeds
    if not seeds:
        if samples: #sampling with replacement to avoid memory issues with high combinations
            seeds=[]
            while len(seeds)<samples:
                ls=tuple(sorted([random.choice(list(sunits)) for j in range(s)])) #use a tuple to be consistent
                if len(set(ls))==s:
                    seeds.append(ls)
            #seeds=[sorted([random.choice(list(sunits)) for j in range(s)]) for i in range(samples)]
        else:
            seeds=list(combinations(sunits,s)) #[['en-1']] or list(combinations(sunits,s)) for example
    if verbose: print('seeds: '+str(len(seeds)))
    for seed in seeds:
        modules[str(seed)] = set([])
        translator[str(seed)] = seed #map between the string and the actual seed numbers

    #iterate through given seeds
    for seed in seeds:
        continue
        #set the starting seed
        act_units=to_list(seed,sunit_map,translator) #list of activated s-unit names
        if names=='string':
            act_nodes={sunit[:-2] for sunit in act_units}
        else:
            act_nodes={int(sunit[:-2]) for sunit in act_units}
        act_prob = {i: {node.name: p for node in N.nodes} for i in range(iterations+1)}
        #print seed, act_units
        for sunit in act_units:
            if names=='string':
                var=sunit[:-2]
            else:
                var=int(sunit[:-2])
            if sunit[-1]=='0': #MODIFIED, this was causing an issue with variable names like 'cdk-0'
                act_prob[0][var]=0.0
            elif sunit[-1]=='1':
                act_prob[0][var]=1.0
                
        #iterate through the mean-field approximation
        for i in range(1,iterations+1):
            for node in N.nodes:
                if node.name in act_nodes and pin_start: #pin the starting nodes
                    act_prob[i][node.name]=act_prob[i-1][node.name]
                    continue
                s,t=mf_approx(node,act_prob,i,reduced=reduced,ds=ds) #solve equation based on probabilities of inputs
                if s>1: s=1.0 #fix rounding errors
                if s<0: s=0.0 #fix rounding errors
                #allow pinning
                if (0 in pinning) and act_prob[i-1][node.name]==0: act_prob[i][node.name]=0.0
                elif (1 in pinning) and act_prob[i-1][node.name]==1: act_prob[i][node.name]=1.0
                else:
                    act_prob[i][node.name]=s
                if i==iterations: 
                    pass
                    #print node.name,act_prob[i][node.name],t,2**node.k,node.inputs

    for seed in seeds: #ALTERNATE
        act_prob=average_seed_mf(N,seed,sunit_map,translator,p,iterations,pinning,reduced,ds,pin_start,
                          update,order,regenerate,oh=False,models=models,names=names)
        i=iterations
        
        if data:
            modules[str(seed)]=act_prob
            continue
        #extract the module set, checks the final iteration
        for node in N.nodes: 
            if act_prob[i][node.name] < 0+tau:  
                #length+=1 #assume these nodes are known
                modules[str(seed)].add(node.name+'-0')
            if act_prob[i][node.name] > 1-tau:
                #length+=1 #assume these nodes are known
                modules[str(seed)].add(node.name+'-1')

    return modules,translator

