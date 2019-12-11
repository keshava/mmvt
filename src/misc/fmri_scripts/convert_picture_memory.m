clear all
hit=0;
miss=0;
fa=0;
cr=0;
HIT_RT=[];
Miss_RT=[];
FA_RT=[];
CR_RT=[];
idx_hit=[];
idx_miss=[];
idx_fa=[];
idx_cr=[];

proj_dir = '/space/megraid/clinical/MEG-MRI/seder/freesurfer/'
% fs_dir = '/cluster/neuromind/mhibert/clinical/temp_analysis/freesurfer/'
subj = 'nmr01287'
subj_dir = [proj_dir, subj]
% subj_fs_dir = [fs_dir, subj]
par_dir = [subj_dir, '/par/']

cd(par_dir)

for j=1:2
    
    % read the behavioral file
respfid=fopen(['retrieve_mem',num2str(j),'_resp.txt'], 'r+t');
resp=textscan(respfid,'%s');
abstime=(resp{1}(1:6:end));
reacttime=resp{1}(2:6:end);
subres=resp{1}(3:6:end);
event=resp{1}(4:6:end);


% read the parameter file
  

  parfid=fopen(['/space/megraid/80/MEG-clin/naoro/MemoryCode/langmempad-00',num2str(j),'.par'], 'r+t');
 %parfid=fopen(['BEHAVIORAL/ex3-00',num2str(j),'.par'], 'r+t');
  param=textscan(parfid,'%s');
  plantime=param{1}(1:5:end);
  for k=1:length(plantime)
      plantimevalue(k)=str2num(plantime{k});
  end
  
  
  planevent=param{1}(2:5:end);
  
  modparam=param; % this will store the new parameter file
  
  
  for i=1:length(abstime)
      idx=find(plantimevalue==floor(str2num(abstime{i})));
      
      if (planevent{idx}=='1'&event{i}=='new')|(planevent{idx}=='2'&event{i}=='old')
      
%           plantimevalue(idx)
%           abstime{i}
%           idx
%           planevent{idx}
%           event{idx}
          disp('ERROR!!!! the parameter file and the response file do  not match, check the time of the trigger!');
          break;
      end
      
      
      
  

       if (planevent{idx}=='1'&subres{i}(1)=='1')
           hit=hit+1;
           modparam{1}(2+5*(idx-1))={'1'} ; % hit mem
           HIT_RT(hit)=str2num(reacttime{i});
       elseif (planevent{idx}=='1'&subres{i}(1)=='2')
          miss=miss+1;
                     modparam{1}(1+5*(idx-1):5*idx)={[]} ; % miss mem
          
                     Miss_RT(miss)=str2num(reacttime{i});
          elseif (planevent{idx}=='2'&subres{i}(1)=='1')
              fa=fa+1;
              %modparam{1}(2+5*(idx-1))={'0'} ; % false alarm mem
               modparam{1}(1+5*(idx-1):5*idx)={[]} ;
               FA_RT(fa)=str2num(reacttime{i});
       elseif (planevent{idx}=='2'&subres{i}(1)=='2')
     cr=cr+1;
              modparam{1}(2+5*(idx-1))={'2'} ; % correct reject
              CR_RT(cr)=str2num(reacttime{i});
  
       end
  
       
       
       
       
       


  end
 modfid=fopen(['mem0',num2str(j),'.par'], 'w+t');
 
 for k=2:length(planevent)  % get rid of the first 4 TRs
     
     n=modparam{1}{(k-1)*5+1}
     if ~isempty(n)
     
  fprintf(modfid,'%s\t%s\t%s\t%s\t%s\n', modparam{1}{(k-1)*5+1:k*5});
     end
     
 end
  

 

end

 hint='This is the behavioral results of pictures'

hit
cr
fa
miss


mean(HIT_RT)
mean(Miss_RT)
mean(FA_RT)
mean(CR_RT)

behave_fid=fopen('Performance.txt', 'w+t');
fprintf(behave_fid, 'Pics Behavioral Statistics: HIT=%d\t CR=%d \t FA=%d\t Miss=%d \t\n', hit*2, cr*2,fa*2, miss*2);
fprintf(behave_fid, 'Pics Reaction Time: HIT=%f\t CR=%f \t FA=%f\t Miss=%f \t',mean(HIT_RT),mean(Miss_RT),mean(FA_RT),mean(CR_RT));