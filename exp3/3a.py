import math
import pandas as pd
data=pd.read_csv('Average.csv')
counts=data.groupby(['Team1','Team1_Status']).size().unstack(fill_value=0)
counts1=data.groupby(['Team2','Team1_Status']).size().unstack(fill_value=0)
counts2=data.groupby(['Venue','Team1_Status']).size().unstack(fill_value=0)

status_count=counts.sum(axis=0)


p_yes=counts['Yes']/status_count['Yes']
p_no=counts['No']/status_count['No']

p_yes1=counts1['Yes']/status_count['Yes']
p_no1=counts1['No']/status_count['No']

p_yes2=counts2['Yes']/status_count['Yes']
p_no2=counts2['No']/status_count['No']

p_yestotal=status_count['Yes']/(status_count['Yes']+status_count['No'])
p_nototal=status_count['No']/(status_count['Yes']+status_count['No'])

result=pd.DataFrame({'Team1':counts.index,'Yes':counts['Yes'],'No':counts['No'],
                     'Total_yes':status_count['Yes'],'Totoal_no':status_count['No'],
                     'P(Yes|Team1)':p_yes,'P(No|Team1)':p_no})
result=result.reset_index(drop=True)

result1=pd.DataFrame({'Team2':counts1.index,'Yes':counts1['Yes'],'No':counts1['No'],
                     'Total_yes':status_count['Yes'],'Totoal_no':status_count['No'],
                     'P(Yes|Team2)':p_yes1,'P(No|Team2)':p_no1})
result1=result1.reset_index(drop=True)

result2=pd.DataFrame({'Venue':counts2.index,'Yes':counts2['Yes'],'No':counts2['No'],
                     'Total_yes':status_count['Yes'],'Totoal_no':status_count['No'],
                     'P(Yes|Venue)':p_yes2,'P(No|Venue)':p_no2})
result2=result2.reset_index(drop=True)

print(result)
print(result1)
print(result2)

Team1=input('Enter Team1 name: ')
Team2=input('Enter Team2 name: ')
Venue=input('Enter Venue: ')

res_filtered=result[(result['Team1']==Team1)]
res1_filtered=result1[(result1['Team2']==Team2)]
res2_filtered=result2[(result2['Venue']==Venue)]

p_yesTeam1=res_filtered['P(Yes|Team1)'].values[0]
p_yesTeam2=res1_filtered['P(Yes|Team2)'].values[0]
p_yesVenue=res2_filtered['P(Yes|Venue)'].values[0]

p_noTeam1=res_filtered['P(No|Team1)'].values[0]
p_noTeam2=res1_filtered['P(No|Team2)'].values[0]
p_noVenue=res2_filtered['P(No|Venue)'].values[0]


p_yesfind=p_yesTeam1*p_yesTeam2*p_yesVenue*p_yestotal
p_nofind=p_noTeam1*p_noTeam2*p_noVenue*p_nototal
total = p_yesfind+p_nofind
print("Total: ",total)
while(math.isclose(total,1.0,abs_tol=0.001)==False):
    print("Normalization")
    p_yesfind=p_yesfind/total
    print("P(Yes|Find):",p_yesfind)
    p_nofind=p_nofind/total
    print("P(No|Find):",p_nofind)
    total=p_yesfind+p_nofind
    print('Total: ',total)

if(p_yesfind>=p_nofind):
    print("Team 1 can win")
else:
    print("Team 1 will lose")  