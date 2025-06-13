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

#국가별 분석 출력
for name, co2, gdp in zip(country_names_list, co2_values, gdp_values):
    trend = "증가" if co2 > 0 else "감소"
    print(f"{name}: CO₂ 배출량은 {trend}했고, GDP 성장률은 {gdp}%였습니다.")


#LLM
import requests
import json

API_KEY = "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"  

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 대화 내용 정자하는 message 리스트 생성하기
messages = [
    {"role": "system", "content": "너는 데이터 분석을 도울 수 있는 AI 모델이야. 사용자 질문에 대한 분석과 인사이트를 제공해."}
]

#사용자 메시지를 이용한 llm과의 대화
def chat_with_llm(user_input):
  
    messages.append({"role": "user", "content": user_input})

    try:
        response = requests.post(url, headers=headers, data=json.dumps({
            "model": "meta-llama/llama-3.3-8b-instruct:free",
            "messages": messages,
        }))
        
        if response.status_code == 200:
            res_dict = response.json()
            reply = res_dict['choices'][0]['message']['content']
            messages.append({"role": "assistant", "content": reply})
            return reply
        else:
            return f"Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        return f"Request Error: {e}"

# 대화 시작
print("LLM과 대화를 시작합니다. 'exit'을 입력하면 종료됩니다.")

while True:
    user_input = input("메세지를 입력하세요요: ").strip().lower()
    if user_input == "exit":
        print("대화를 종료합니다.")
        break
    
    response_text = chat_with_llm(user_input)
    print(response_text)

#선형 회귀 방정식_과거 데이터를 통한 미래 예측
from sklearn.linear_model import LinearRegression
import numpy as np

# 선형 회귀 학습
model = LinearRegression()
model.fit(np.array(co2_values).reshape(-1, 1), gdp_values)

# 회귀선 그리기-일반적인 회귀모델 사용
x_line = np.linspace(min(co2_values), max(co2_values), 100)
y_line = model.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, color='gray', linestyle='--', label='회귀선')

print(f"회귀식: GDP = {model.coef_[0]:.2f} * CO2 + {model.intercept_:.2f}")
