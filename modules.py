#Module computations for general threshold networks
#Thomas Parmer, 2022

import pandas as pd
from cana.utils import statenum_to_binstate, binstate_to_statenum
from collections import deque
import random
from itertools import combinations

#alternate version of CANA LUT table function that fixes bug, takes a BooleanNode
#NOTE: this first tries to read the look-up-table from an existing data structure, otherwise it calculates it
def look_up_table(node,ds={}):
    if ds:
        return ds[node.name]
    
    d = []
    for statenum, output in zip( xrange(2**node.k), node.outputs):
        # Binary State, Transition
        d.append( (statenum_to_binstate(statenum, base=node.k), output) )
    df = pd.DataFrame(d, columns=['In:','Out:'])
    return df


#helper functions for GTN modules
#TODO: convert this to work with RBNs
#NOTE: some networks have s-units T-0/T-1, etc; the convention here is that T- or F- with an _ later is a t-unit or fusion unit
#additionally, an s-unit must end with a digit (state) and 'F-T' indicates fusion nodes created in the threshold network
def is_sunit(s):
    """ determines if a particular unit is an s-unit based on the name """
    if not s or len(s)<3 or not s[-1].isdigit() or s[:3]=='F-T' or ((s[:2]=='F-' or s[:2]=='T-') and '_' in s):
        return False
    return True

def reduce_step(step):
    """ reduces a module step to only include s-units """
    new_step={s for s in step if is_sunit(s)}
    return new_step

def reduce_module(module):
    """ reduces a module to only include s-units """
    new_module={t: {s for s in module[t] if is_sunit(s)} for t in module}
    return new_module

def extract_sunits(module,pinned=True):
    """ extracts the unique s-units in a DCM module """
    module=reduce_module(module)
    if pinned:
        units={s for t in module for s in module[t]}
    else:
        units={s for s in module[max(module)]} #extract only the last time step
    return units

def extract_variables(module,pinned=True):
    units=extract_sunits(module,pinned=pinned)
    return {s[:-2] for s in units} #remove the state and force uniqueness of variables

def count_sunits(module,pinned=True):
    """ counts the unique s-units in a DCM module """
    units=extract_sunits(module,pinned=pinned)
    return len(units)

def count_variables(module,pinned=True):
    return len(extract_variables(module,pinned=pinned))

def sunit_to_var(s):
    """ extract variable from s-unit name """
    return s[:-2]


#manual neighbors function for cana; takes BooleanNetwork, BooleanNode as inputs
def neighbors_BN(n,node):
    return [x.name for x in n.nodes if node.name in x.inputs]


###General Threshold Network functions
#threshold network conversion from a DCM
def create_threshold_network(n):
    """ Create a digraph compatible with threshold network manipulation
    Expects a DCM-like digraph as input, returns a threshold-like digraph where every node is a literal and/or threshold node
    
    Valid nodes include these properties: label, threshold (tau), type, time delay, variable, state
    Valid edges include these properties: type 
    
    NOTE: the DCM originally has a separate fusion node for each literal enput of the same state involved in the same symmetry group
    However, this means that the same enput can activate multiple edges rather than just one
    As a solution, each group of fusion nodes that are associated with the same state and the same symmetry group are replaced
    by one threshold node with tau=n, where n is the number of fusion nodes that were combined into that threshold node """
    
    n=n.copy() #don't change the original graph
    
    #check for similar fusion nodes
    fusion_nodes=[name for name in n.nodes() if n.node[name]['type']=='fusion']
    groups={} #each group will share the same predecessors and successors
    i=0
    for f in fusion_nodes:
        predecessors=set(n.predecessors(f))
        successors=set(n[f])
        found=False
        for key in groups:
            if groups[key]['predecessors']==predecessors and groups[key]['successors']==successors: #we have a match
                groups[key]['nodes'].append(f)
                found=True
                break
        if not found: #make a new group
            groups[i]={'predecessors':predecessors, 'successors':successors, 'nodes':[f]}
            i+=1
    #delete fusion nodes and replace with new threshold nodes
    for t in groups: #add new threshold nodes
        name="F-T"+str(t)
        tau=len(groups[t]['nodes'])
        group=n.node[random.choice(tuple(groups[t]['successors']))]['group'] #take group attribute of random successor, necessary for visualization
        n.add_node(name,{'label':str(tau),'type':'threshold','tau':tau,'group':group})
        #adjust successor node thresholds; add 1 for new threshold node and then substract for each fusion node deleted
        for successor in groups[t]['successors']: #there should only be one
            n.add_edge(name,successor)
            n.node[successor]['tau'] += 1 - tau
            n.node[successor]['label'] = str(n.node[successor]['tau'])
        #add new edges to threshold nodes from predecessors
        for predecessor in groups[t]['predecessors']:
            n.add_edge(predecessor,name)
        
    #delete fusion nodes and adjacent edges
    for f in fusion_nodes:
        n.remove_node(f)
    #print groups
    
    for name in n.nodes():
        
        if 'label' not in n.node[name]:
            n.node[name]['label']='unknown'
            
        if 'type' not in n.node[name]:
            n.node[name]['type']='unknown'
            
        if 'tau' not in n.node[name]:
            if n.node[name]['type']=='threshold':
                n.node[name]['tau']=1
            else:
                n.node[name]['tau']=0
                
        if 'delay' not in n.node[name]:
            if n.node[name]['type']=='threshold':
                n.node[name]['delay']=0
            else:
                n.node[name]['delay']=1
        
        #assume variable name comes from the label
        if 'variable' not in n.node[name] and n.node[name]['type']=='variable':
            n.node[name]['variable']=n.node[name]['label']
            
        #assume states are only positive integers
        if 'state' not in n.node[name] and n.node[name]['type']=='variable':
            state_str=name.replace(n.node[name]['variable'],"") #remove variable name
            state_str=''.join([s for s in state_str if s.isdigit()]) #concatenate string from all integers left in name (in case of double-digit states)
            n.node[name]['state']=int(state_str)
                
        #print name,n.node[name]
        
    return n


#specialized BFS to work for threshold network
def BFS_threshold(n,seed,input_type='steady',time_limit=10,conversion=True,pinned=set([])):
    
    """ Does specialized BFS on a threshold network n
    input_type may be 'pulse' if a single signal is sent through the network (seed guaranteed ON/OFF for only one time step)
    or 'steady' if the signal is held on (seed guaranteed ON/OFF for all time steps) 
    seed may be a single node or a list of starting nodes
    
    Implementation based on BFS with a priority queue of times {t: q} 
    where t is the time when those nodes become active and q are the nodes active at that time step 
    
    Returns a dictionary of nodes visited at each time step
    
    NOTE: a threshold network is required, this will break if fusion nodes have not been removed
    Assume that a conversion is needed; will not run conversion if conversion argument set to False 
    
    NOTE: if run with input_type='steady', unfolding contains only the new states visited at that time step
    while all previous states visited are also considered active
    if run with input_type='pulse', unfolding contains EVERY node that is active at that time step, 
    even if it has been active before
    the pinned argument is used to specify nodes that are pinned in one state, regardless of the input_type,
    and can specify a subset of the seed that's constitutively active when input_type='pulse'
    
    NOTE: if run with input_type='steady', then ignore any node that is a different state of the starting seed;
    because we are pinning control of that variable, we need to remove all other possible variable states from the network """
    
    if conversion: #note conversion is slow due to a network deep copy
        try:
            n=create_threshold_network(n)
        except:
            raise Exception("Network could not be converted to a thresholded representation")
        
    if not isinstance(seed,list): #convert seed to a list if it is a single node
        seed=[seed]
    
    for node in seed:
        if node not in n.nodes():
            raise Exception('Seed not in the network!  Hint: did you specify both the node and current state? e.g. wg-1')
            
    #if there is steady-state input (pinning control), note any contradictions to any node in the seed
    #NOTE: if given a logical contradiction, this function picks only ONE of the states to use
    #however, this doesn't affect steady-state (because the variable was already added to the queue and active step)
    if input_type=='steady':
        vals={n.node[x]['variable']:n.node[x]['state'] for x in seed if n.node[x]['type']=='variable'}
        contradictory_nodes={node for node in n if 
                        n.node[node]['type']=='variable' and n.node[node]['variable'] in vals and n.node[node]['state']!=vals[n.node[node]['variable']]}
    else:
        contradictory_nodes=set([])
    if pinned: #check pinned nodes as well
        vals={n.node[x]['variable']:n.node[x]['state'] for x in pinned if n.node[x]['type']=='variable'}
        contradictory_nodes.update({node for node in n if 
                        n.node[node]['type']=='variable' and n.node[node]['variable'] in vals and n.node[node]['state']!=vals[n.node[node]['variable']]})
        
    counter=0  #this will increment towards the time limit and provide an exit if we get stuck
    visited=set() #visited nodes during the entire history
    thresholds={} #global threshold dictionary to populate with discounted threshold nodes
        
    #create 'priority queue', actually a dictionary of time steps due to difficulty in changing priorities in a heap
    time_steps={0: deque(seed)} #FIFO queue for each time step of nodes to attempt to visit
    time=0 #global chronometer, iterates by one as dynamic unfolding takes place
    unfolding={} #track the unfolding of the dynamics over iterated time, mirrors time_steps but only holds visited nodes    
    active_step=set(seed) #active nodes for this time step, will be different than visited if input_type='pulse'
    for node in pinned:
        time_steps[0].append(node)
        active_step.add(node)
    
    
    while time<=max(time_steps): #allows for possibility of time jumps, i.e. updates further ahead than one time step
        
        #update counter
        old_length=len(visited)
        visited.update(active_step)
        #if len(visited)==old_length: #increase counter if we have not added anything new
        #if any(unfolding[t] == active_step for t in unfolding): #increase counter if we repeat a past partial configuration
        counter+=1
        #print time,len(visited)
        #reset thresholds and active_step if this is pulse (so that we may revisit nodes, including the starting nodes)
        #print time,active_step
        if input_type=='pulse':
            thresholds={}
            active_step=set()
            #add in pinned nodes first
            for node in pinned:
                time_steps.setdefault(time+1,deque())
                time_steps[time+1].append(node)
                active_step.add(node)
        
        #check for exit due to being stuck in a cycle
        if counter>=time_limit: 
            break
            

        #run BFS
        while time_steps.get(time): 
            node=time_steps[time].popleft()
            #add node to unfolding
            unfolding.setdefault(time,set())
            unfolding[time].add(node)
            
            #check neighbors - note that for steady state we assume all past actors are still activated, but pulse requires repeat activation
            for neighbor in n[node]:
                #ignore any contradictory nodes; better to ignore than remove because we don't want BFS to be destructive to the graph
                if neighbor in contradictory_nodes:
                    continue
                if neighbor not in active_step:
                    #check that neighbor passes the threshold test if it has a non-trivial threshold
                    if n.node[neighbor]['tau']>1:
                        thresholds.setdefault(neighbor,set())
                        thresholds[neighbor].add(node)  #add node due to this incoming edge, ensures node can only contribute once
                    if neighbor not in thresholds or len(thresholds[neighbor])>=n.node[neighbor]['tau']:
                        #at this point we have passed the threshold, so insert neighbor at proper time slot t
                        t = time + n.node[neighbor]['delay']
                        time_steps.setdefault(t,deque())
                        time_steps[t].append(neighbor)
                        active_step.add(neighbor)
                           
        #next iteration
        time+=1
    
    return unfolding


#function to find top modules given a set of modules
#runs a naive check that doesn't account for destructive interference
def find_top_modules(nu,tm,sizes,candidates,input_type='steady'):
    """ Determines the top modules by seeing which modules can be subsumed into other modules
    Specifically, it tests every candidate to see if it is along the path of a larger module in nu, O(n**2) time 
    Candidates is a dictionary where the keys are node names and the values are actual nodes to check """
    
    nodes_seen=[] #track which nodes the candidate might be found in
    for node in sorted(sizes,key=lambda x:sizes[x],reverse=True): #start with the largest modules

        if node in candidates: #only check our candidates
            
            seen=False
            for step in nodes_seen: #compare against all dynamic steps we've seen before 
                
                if not candidates[node]-step: #the candidate is a subset of the nodes seen
                    #print node,"eaten by",step
                    seen=True
                    break
            
            if not seen:
                #add to top_modules
                tm[node]=nu[node]
                #add the node's unfolding to nodes_seen
                if input_type=='steady':
                    nodes_seen.append({x for t in nu[node] for x in nu[node][t]})
                else: #add a separate set for each time step
                    for t in nu[node]:
                        nodes_seen.append({x for x in nu[node][t]})
                        
            
    return tm


#function to find all dynamic modules of single seed perturbations
def find_dynamic_modules(n,num=1,input_type='steady',time_limit=10,conversion=True,samples=None,tm=True,seeds=None):
    
    """ Test all possible seeds of a network to determine each seed's dynamic unfolding
    If tm=True, combines any unfolding that is a subset of another unfolding and returns the largest unique unfoldings 
    
    input_type may be 'pulse' if a single signal is sent through the network (seed guaranteed ON/OFF for only one time step)
    or 'steady' if the signal is held on (seed guaranteed ON/OFF for all time steps) 
    
    num dictates how many nodes the seed should contain; the default is to only search dynamic modules from one starting node
    NOTE: this function does a combinatoric search over all possible seed configurations, be careful with memory constraints
    when running with high n
    If samples=x, then this will draw x samples from the possible combinations with replacement
    If tm=True, this will try to compute all top modules
    If seeds is not None, this will find all dynamic modules from the seeds contained in that iterable (must be hashable);
    Note that seeds takes precedence over samples
    
    Converts the network n to a thresholded representation """
    
    if conversion:
        try:
                n=create_threshold_network(n)
        except:
                raise Exception("Network could not be converted to a thresholded representation")
    
    network_unfolding,sizes={},{}
    #remove all non-variable nodes for cleaner visibility and comparison
    nodes=[x for x in n if n.node[x]['type']=='variable']

    if not seeds:
        if samples: #sampling to avoid memory issues with high combinations
            seeds=[[random.choice(nodes) for j in range(num)] for i in range(samples)]
        else: #get all combinations
            seeds=combinations(nodes,num)
    #elif num>1:
    #    #find combinations of our starting seeds; may need to be explicitly passed 
    #    seeds=combinations(seeds,num)
    candidates={} #used for top modules later
    #print seeds

    for seed in seeds:
        seed=list(seed)
        #quick check for contradiction
        vals=[n.node[x]['variable'] for x in seed]
        if len(set(vals))<len(vals): #we have multiple nodes sharing the same variable
            continue
        
        unfolding=BFS_threshold(n,seed,input_type=input_type,time_limit=time_limit,conversion=False)
        unfolding={t:{x for x in unfolding[t] if n.node[x]['type']=='variable'} for t in unfolding} #simplify
        #if num==1: 
        #    seed=seed[0] #special case for single nodes, allows us to find top modules
        network_unfolding[frozenset(seed)]=unfolding #UPDATED to frozenset
        sizes[frozenset(seed)]=sum([len(unfolding[t]) for t in unfolding])
        #update candidates
        candidates[frozenset(seed)]=set(seed)
        #if tm: candidates[str(seed)]=set(seed)
 
    #condense down to the top modules by finding which modules are subsets of other modules (only consider variables)
    #NOTE: if a node is turned on at any point, it is guaranteed to follow the trajectory specified by its own module
    #a node along the path of another module can therefore have its trajectory be subsumed into that module (assuming no contradiction)
    top_modules={}
    if tm:
        top_modules=find_top_modules(network_unfolding,top_modules,sizes,candidates,input_type=input_type)
    
    
        
    return network_unfolding,top_modules,sizes,candidates

