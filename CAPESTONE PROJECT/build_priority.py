import pandas as pd, numpy as np, json
L=pd.read_csv('leaf_topics_labeled.csv'); S=pd.read_csv('hierarchy_stability.csv'); G=pd.read_csv('onet_gap_analysis.csv')
O=pd.read_csv('onet_evaluation.csv'); E=pd.read_csv('leaf_evidence.csv')
d=L[['topic_id','topic_name','level1_name','level2_name','topic_size','label_confidence','mixed_topic','coherent','critic_action']].merge(
  S[['topic_id','level1_parent_persistence','level2_parent_persistence']],on='topic_id').merge(
  G[['topic_id','coverage_class','purity','nearest_neighbor_agreement','semantic_alignment']],on='topic_id').merge(
  E[['topic_id','boundary_mean_sim','mean_sim_to_centroid']],on='topic_id')
p25_pur=d.purity.quantile(.25); p25_conf=d.label_confidence.quantile(.25); p25_bnd=d.boundary_mean_sim.quantile(.25)
sc=[];why=[]
for _,r in d.iterrows():
    s=0;w=[]
    if r.mixed_topic: s+=3;w.append('Gemini flagged mixed_topic')
    if not r.coherent: s+=1;w.append('label marked not coherent')
    if r.label_confidence<=p25_conf and r.label_confidence<0.9: s+=1.5;w.append(f'low label confidence ({r.label_confidence:.2f})')
    if r.level1_parent_persistence<0.6: s+=2;w.append(f'Level-1 persistence {r.level1_parent_persistence:.1f}')
    if r.level2_parent_persistence<0.6: s+=2;w.append(f'Level-2 persistence {r.level2_parent_persistence:.1f}')
    s+=(1-r.level1_parent_persistence)+(1-r.level2_parent_persistence)
    if r.semantic_alignment=='NO_MATCH': s+=3;w.append('Gemini O*NET judgment NO_MATCH')
    elif r.semantic_alignment=='PARTIAL_MATCH': s+=1;w.append('PARTIAL_MATCH to O*NET')
    if r.coverage_class=='POSSIBLE_EMERGING_WORK': s+=3;w.append('possible emerging work (limited O*NET alignment)')
    elif r.coverage_class=='PARTIALLY_COVERED': s+=0.5
    if r.purity<=p25_pur: s+=1;w.append(f'low O*NET purity ({r.purity:.3f})')
    if r.boundary_mean_sim<=p25_bnd: s+=1;w.append(f'weak boundary tasks (mean cos {r.boundary_mean_sim:.3f})')
    if r.critic_action not in ('ACCEPT','NOT_REVIEWED'): s+=2;w.append(f'critic action {r.critic_action}')
    sc.append(round(s,2)); why.append('; '.join(w) or 'no individual risk flag; review for completeness')
d['risk_score']=sc; d['risk_reason']=why
d=d.sort_values(['risk_score','topic_size'],ascending=[False,False]).reset_index(drop=True)
d['review_priority']=np.arange(1,81)
out=d.rename(columns={'level1_name':'level1_domain','level2_name':'level2_family','level1_parent_persistence':'level1_persistence','level2_parent_persistence':'level2_persistence','semantic_alignment':'onet_alignment'})
cols=['review_priority','topic_id','topic_name','level1_domain','level2_family','topic_size','label_confidence','level1_persistence','level2_persistence','onet_alignment','purity','risk_reason']
out[cols+['risk_score']].to_csv('human_review_priority.csv',index=False)
print(out[cols[:3]+['risk_score']].head(10)); print('P25 thresholds',p25_pur,p25_conf,p25_bnd)
