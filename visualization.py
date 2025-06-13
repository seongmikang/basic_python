import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('2023_data.csv')

# LULUCF 제외 CO2 배출량 row들 뽑아내기
Rows_CO2_emission_change = data[data['Series Name'] == 'Carbon dioxide (CO2) emissions (total) excluding LULUCF (% change from 1990)']

# 연간 gdp 성장률 row들 뽑아내기
Rows_GDP_change = data[data['Series Name'] == "GDP growth (annual %)"]

# 36개의 국가 데이터 중, 랜덤으로 10개의 국가를 뽑기
countries = Rows_CO2_emission_change['Country Name']
random_countries = countries.sample(10)

random_countries_CO2 = Rows_CO2_emission_change[Rows_CO2_emission_change["Country Name"].isin(random_countries)]

random_countries_GDP = Rows_GDP_change[Rows_GDP_change["Country Name"].isin(random_countries)]

# 그 해당 값들만 뽑아오기
CO2_emission_change = random_countries_CO2[["Country Name","2023 [YR2023]"]]

GDP_change = random_countries_GDP[["Country Name","2023 [YR2023]"]]

# 인터넷 도움 : 두 값을 한 행에 놓기 = merge
merged_data = pd.merge(CO2_emission_change, GDP_change ,on='Country Name', suffixes=('_CO2','_GDP'))

# 인터넷 도움 : merged_data 컬럼의 값들을 각각 리스트로 뽑아내기 = .tolist()
country_names_list = merged_data['Country Name'].tolist()
co2_tolist = merged_data['2023 [YR2023]_CO2'].tolist()
gdp_tolist = merged_data['2023 [YR2023]_GDP'].tolist()


# 리스트값 실수 변환 + 자릿수 제어
co2_values = []
for i in co2_tolist:
    co2_values.append(int(float(i)*100)/100)
# print(co2_values)

gdp_values = []
for j in gdp_tolist:
    gdp_values.append(int(float(j)*10000)/10000)
# print(gdp_values)


# 표 그리기 -> x,y축 주석 사이즈 조정
fig, ax = plt.subplots(figsize = (12,10))

#색깔 지정
point_colors = ['blue','green','red','cyan','magenta','yellow','black','lime','orange','purple']
color_map = plt.cm.get_cmap('plasma')
for i in range(len(co2_values)):
    ax.scatter(co2_values[i], gdp_values[i]
           , color = point_colors[i], label = country_names_list[i],edgecolors='black')

# x,y축 값 설정하기
ax.set_xlim(min(co2_values)-5, max(co2_values)+10) 
ax.set_ylim(min(gdp_values)-2, max(gdp_values)+2) 

# x = 0 , y = 0 점선 그리기
ax.axvline(0, color='gray', linestyle='--', linewidth=1.5) 
ax.axhline(0, color='gray', linestyle='--', linewidth=1.5) 


ax.set_xlabel('Carbon dioxide (CO2) emissions (total) excluding LULUCF (% change from 1990)', fontsize=12) 
ax.set_ylabel('GDP growth (annual %)', fontsize=12) 
ax.set_title('GDP growth vs CO2 emissions(% change from 1990) (2023)', fontsize=15) 
ax.tick_params(labelsize = 10)

plt.legend(fontsize = 5 , markerscale = 0.6)
plt.xticks(rotation=45)
plt.show()