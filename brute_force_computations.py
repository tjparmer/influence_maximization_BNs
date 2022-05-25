#Brute-force computations
#Thomas Parmer, 2022

#from cana.utils import statenum_to_binstate, binstate_to_statenum
from modules import *


#reverse sunit_map (or any dictionary)
def reverse_sunit_map(sunit_map):
    #print sunit_map
    return {sunit_map[key]:key for key in sunit_map}


#determine ground-truth driver sets based on brute-force computation
#reverse variable:s-unit lookup
def get_variable_map(N,sunit_map):
    var_map={node.name:[] for node in N.nodes}
    for s in sunit_map:
        if sunit_map[s][:-2] not in var_map: #.split("-")[0] not in var_map:
            raise KeyError("Variable Name invalid:",sunit_map[s].split("-")[0])
        var_map[sunit_map[s][:-2]].append(s) #.split("-")[0]].append(s)
    return var_map


#create function to only consider subsets of free variables (exponential rather than combinatoric search)
def exponential_configs(N,sunits,sunit_map,pinned_sunits,free_vars=None):
    pinned_vars={sunit_map[x][:-2] for x in pinned_sunits} #.split("-")[0] for x in pinned_sunits}
    var_map=get_variable_map(N,sunit_map)
    if not free_vars: #assume all other variables are free
        free_vars=set(var_map.keys())-pinned_vars
    #print pinned_vars,free_vars
    #implement an exponential search over all combinations of free_vars
    configs=[]
    for statenum in xrange(2**len(free_vars)): #ASSUMES BINARY STATES
        configs.append(statenum_to_binstate(statenum, base=len(free_vars)))
    configs=[[var_map[var][int(binstate[j])] for j,var in enumerate(free_vars)] for i,binstate in enumerate(configs)]
    for config in configs:
        config.extend(list(pinned_sunits)) #include the pinned variables in the config
    #print [[sunit_map[s] for s in x] for x in configs]
    return configs


#translate a config (list of s-units) into {node: value} dictionary
def translate_config_to_vars(config,N,sunit_map):
    seed=[sunit_map[s] for s in config]
    if len(seed)!=len(N.nodes):
        #print "please provide one value for all variables"
        return
    config={node:'#' for node in N.nodes}
    for s in seed:
        for node in N.nodes:
            if node.name==s[:-2]: #s.split("-")[0]:
                config[node]=s.split("-")[-1] #s.split("-")[1]
    for n in config:
        if config[n]=='#':
            print("could not resolve all variable states in config!")
    return config


#return a config {BooleanNode: state} from a list of sunits
def config_from_sunits(N,seed,sunit_map):
    config=[reverse_sunit_map(sunit_map)[s] for s in seed]
    return translate_config_to_vars(config,N,sunit_map)


#iterate config dictionary based on LUTs (synchronous)
def network_step(config,pinned_vars={}):
    node_dic={node.name: node for node in config} #lookup node object by name
    new_step={node:config[node] for node in config} #NOTE: don't use a deepcopy or it will duplicate keys
    #print len(new_step),len(config)
    for node in config:
        if not node.name in pinned_vars: #otherwise, keep the value the same
            inputs=[str(config[node_dic[x]]) for x in node.inputs]
            input_str=''.join(inputs)
            new_step[node]=node.outputs[binstate_to_statenum(input_str)]
            #print node.name,node.inputs,input_str,new_step[node]#,node.look_up_table()
    #print len(new_step),len(config),{x.name:new_step[x] for x in new_step}
    return new_step


#function to determine the length of an attractor
def attractor_length(diffusion):
    diffusion=[diffusion[t] for t in diffusion]
    cycle=1 #assume no limit cycle
    for i,step in enumerate(diffusion):
        if step in diffusion[:i]:
            cycle=i-diffusion.index(step)
            break
    return cycle


#function for solving transfer functions while allowing for pinned nodes
#starts from an initial configuration and continues until a steady-state or time limit is reached
#returns a diffusion dictionary {t: {set of s-units reached}}
def network_dynamics(config,pinned_vars={},time_limit=10,update='synchronous',break_early=True):
    t=0
    diffusion={0: {x.name+'-'+str(config[x]) for x in config}}
    while t<time_limit:
        if update=='synchronous':
            config=network_step(config,pinned_vars=pinned_vars)
        else: #TODO: provide asynchronous functions
            pass
        t+=1
        new_step={x.name+'-'+str(config[x]) for x in config}
        if new_step==diffusion[t-1] and break_early: break #we have found a steady-state
        diffusion[t]=new_step #else, keep iterating
    return diffusion


#function for solving transfer functions while allowing for pinned nodes and asynchronous update order (deterministic or stochastic)
#starts from an initial configuration and continues until a steady-state or time limit is reached
#returns a diffusion dictionary {t: {set of s-units reached}}
#NOTE: if update is stochastic, then break_early should be False
def network_dynamics_asynchronous(N,config,pinned_vars={},time_limit=10,break_early=True,order=None,
                    regenerate=False,oh=False,replacement=False):
    
    if not order:
        if not replacement: order=[[node] for node in random_node_order(N)] #random by default, sample without replacement
        else: order=[[random.choice(N.nodes)] for node in range(len(N.nodes))] #sample with replacement
    order_history={} #track order to replicate
    t=0
    diffusion={0: {x.name+'-'+str(config[x]) for x in config}}
    while t<time_limit:
        t+=1
        order_history[t]=order
        #print [[node.name for node in block] for block in order]
        #print "Iteration:",t,{x.name:config[x] for x in config}
        for block in order:
            config=network_step(config,pinned_vars=pinned_vars,nodes=block)
        new_step={x.name+'-'+str(config[x]) for x in config}
        if new_step==diffusion[t-1] and break_early: break #we have found a steady-state
        diffusion[t]=new_step #else, keep iterating
        if regenerate:
            if not replacement: order=[[node] for node in random_node_order(N)] #new random order, sample without replacement
            else: order=[[random.choice(N.nodes)] for node in range(len(N.nodes))] #sample with replacement
            
    if oh: return diffusion,order_history
    return diffusion


#run network dynamics for synchronous and deterministic/stochastic asynchronous update schedules
#NOTE: should pass break_early=False for stochastic updates
def run_network_dynamics(N,config,sunit_map,pinned_vars={},time_limit=10,update='synchronous',break_early=True,order=None,
                    regenerate=False,oh=False):
        
    config=config_from_sunits(N,config,sunit_map)
    if update=='synchronous':
        return network_dynamics(config,pinned_vars,time_limit,update=update,break_early=break_early)
    if update=='asynchronous':
        return network_dynamics_asynchronous(N,config,pinned_vars,time_limit,break_early,order,regenerate,oh,replacement=False)
    if update=='stochastic asynchronous':
        return network_dynamics_asynchronous(N,config,pinned_vars,time_limit,break_early,order,regenerate,oh,replacement=True)


#do a brute-force search over (full or partial) configs to determine all (full or partial) steady-states
def find_attractors_bruteforce(N,DCM,sunits,sunit_map,pinned_sunits,free_vars=None,time_limit=10,update='synchronous',verbose=False):
    configs=exponential_configs(N,sunits,sunit_map,pinned_sunits,free_vars)
    pinned=[sunit_map[s] for s in pinned_sunits]
    fixed_points={} #may be full or partial
    solutions=set([])
    lengths={} #holds the length of the found attractors (1 for fixed points, >1 for limit cycles)
    for seed in configs:
        nodes_reached=None
        config=translate_config_to_vars(seed,N,sunit_map)
        seed=[sunit_map[s] for s in seed]
        #print seed,pinned
        if len(seed)==len(N.nodes): #exact computation
            diffusion=network_dynamics(config,pinned_vars={x[:-2] for x in pinned},time_limit=time_limit,update=update) #{x.split("-")[0] for x in pinned}
        else: #rely on partial information computation via the DCM
            diffusion=BFS_threshold(DCM,seed,'pulse',time_limit=time_limit,pinned=pinned) #pin seed for steady-state modules
            diffusion=reduce_module(diffusion) #get rid of t-units
        solutions.add(max(diffusion))
        for t in diffusion:
            if frozenset(diffusion[t]) in fixed_points: #check for a configuration already seen (guarantees future iterations are the same)
                nodes_reached=frozenset(diffusion[t])
        if not nodes_reached:
            nodes_reached=frozenset(extract_sunits(diffusion,pinned=False)) #consider only the last iteration (long-term behavior)
        #print nodes_reached,diffusion
        #if max(diffusion)>12: return diffusion #len(nodes_reached)>12
        fixed_points.setdefault(nodes_reached,0)
        fixed_points[nodes_reached]+=1
        lengths.setdefault(nodes_reached,set([]))
        lengths[nodes_reached].add(attractor_length(diffusion))
    if verbose: 
        print("max time: "+str(max(solutions)))#,solutions #determine how long it takes to find solutions for a given system
        print("fixed point lengths: "+str(lengths))
    return fixed_points


#convert attractors from the STG of the network to a dictionary representation where each attractor is {variable names:ON or OFF}
def attractors_to_dict(n,mode='stg'):
    
    attractor_dict={}
    attractors=n.attractors(mode=mode)
    for attractor in attractors:
        label=n._stg.node[attractor[0]]['label'] #only pay attention to the first state for now
        #for i,var in enumerate(N.nodes): print i,type(var.name),type(label)
        #attractor_dict[label]={var.name:label[i] for i,var in enumerate(N.nodes)}
        if len(attractor)==1: #fixed point
            attractor_dict[label]={var.name+'-'+label[i] for i,var in enumerate(n.nodes)} #alternative, assumes labeling standard
        else: #limit cycle
            label=tuple([n._stg.node[attractor[i]]['label'] for i in range(len(attractor))])
            attractor_dict[label]=[{var.name+'-'+n._stg.node[attractor[a]]['label'][i] for i,var in enumerate(n.nodes)} for a in range(len(attractor))]
    return attractor_dict


#return all input nodes (no inputs except themselves)
def network_inputs(n):
    return [node for node in n.nodes if not node.inputs or (len(node.inputs)==1 and node.inputs[0]==node.name)]


def determine_attractor_inputs(n,sunit_map,attractor,attractors):
    """ return the network's input nodes in the appropriate state for the desired attractor 
    for limit cycles, this chooses the first configuration arbitrarily """
    
    inputs=[node.name for node in network_inputs(n)]
    if type(attractor)==tuple: #limit cycle
        attractor=attractors[attractor][0] #only consider the first configuration as inputs won't change anyways
    else:
        attractor=attractors[attractor]
    rmap=reverse_sunit_map(sunit_map)
    #print attractor
    return tuple(sorted([rmap[sunit] for sunit in attractor if sunit[:-2] in inputs]))

