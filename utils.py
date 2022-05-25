#Utility functions for influence calculations
#Thomas Parmer, 2022

from itertools import combinations

def compute_jaccard(set1,set2):
    """ return the jaccard index of two sets: intersection / union """
    return float(len(set1.intersection(set2))) / len(set1.union(set2))


#list method that ensures that s-unit combinations are in the correct order
#order is based on node number and state, starting with 0
def to_list(seed,smap,translator=None):
    """ determine the sunits involved in the seed, based on the sunit_map 
    this make take either a tuple of s-units, or a string of s-units (requires the translator argument)
    the smap maps individual node numbers to names, the translator maps seed strings to node numbers """
    
    ls = []
    if isinstance(seed,str):
        seed = translator[seed]
        
    for node in seed:
        ls.append(smap[node])
    return ls


#define s-units and modules; map any node set to numbers starting with 0
def get_sunits(N):
    sunits,sunit_map=set([]),{}
    num=0
    for node in N.nodes:
        for state in ['0','1']:
            sunits.add(num)
            sunit_map[num]=str(node.name)+'-'+state
            num+=1
        
    return sunits,sunit_map


#reduce seeds based on length (range inclusive) and contradiction
def reduce_seeds(seeds,sunit_map,translator,lrange=None,length=1):
    new_seeds=set([])
    if not lrange: #NOTE: if lrange is set, length argument is ignored
        lrange=[length,length]
    for seed in seeds: 
        sunits=to_list(str(seed),sunit_map,translator)
        if len(sunits)<lrange[0] or len(sunits)>lrange[1]: #incorrect length
            continue
        if len({sunit[:-2] for sunit in sunits})!=len(sunits): #there is a contradiction
            continue
        new_seeds.add(seed)
        
    return new_seeds


#create translator function if needing a translator but not wanting to run the actual mf-approximation
def create_translator(N,s=1,sunits=None,sunit_map=None,translator={},seeds=None,samples=None):
    """ find all pathway modules seeds for a given network N and seed size s, with the given translator,
    can iteratively add to translator for different s values; returns updated translator
    set seeds to a list of which seeds you want to find modules for (by s-unit number)
    or set samples to sample the space of possible seeds """
    
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
        else:
            seeds=list(combinations(sunits,s)) #[['en-1']] or list(combinations(sunits,s)) for example
    #print 'seeds:',len(seeds)
    for seed in seeds:
        if isinstance(seed,str): seed=eval(seed) #NOTE: eval in case these are strings
        translator[str(seed)] = seed #map between the string and the actual seed numbers
        
    return translator


