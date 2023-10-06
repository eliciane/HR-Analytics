# RH ANALYTICS
Segundo o blog do "Great Place to Work" fatores que levam ao turnover 
de funionários são: (a) contratos de flexíveis de curto prazo e freelance, que geram mais autonomia; 
(b) cultura de trabalho remoto e agenda flexivel; (c) ausencia de oportunidades de progressão na carreira; (d) inadequado salários ou benefícios; (e) insatisfação com a liderança e cultura organizacional
(e) falta de equilibrio entre bem-estar e stress do trabalho.

Este projeto tem o objetivo de analisar os fatores que levam colaboradores a rejeitarem promoções para novas posições/cargos e, consequentemente, possuem mais probabilidade de pedir demissão.

Em razão de já existir uma classificação prévia dos colaboradores, rotulados em “SIM, solicitou demissão” e “NÃO solicitou demissão, o modelo de machine learning a ser desenvolvido será treinado com os dados dos colaboradores para predizer em quais destes rótulos ou classificação os novos colaboradores poderiam se encaixar. Ou seja, o resultado é saber qual a probabilidade de solicitar demissão. Portanto, o algoritmo a ser desenvolvido diz respeito a um modelo supervisionado de classificação, em que temos como resultado a classificação dentro de uma variável/atributo categórica já conhecida previamente. 

Primeiramente, as features foram selecionadas pela técnica de análise de correlação. E então, foram empregados algoritmos de Random Forest e Regressão logística.


A base de dados analisada neste projeto é de um famoso dataset da IBM, que contém 35 variáveis:
[https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset.)

0 - Age: The age of the employee.

1 - Attrition: The target variable indicating whether an employee has attrited (left the company).

2- BusinessTravel: The frequency of business travel.

3 - DailyRate: The daily rate of pay.

4 - Department: The department in the company where the employee works.

5 - DistanceFromHome: The distance from the employee's home to the workplace.

6 - Education: The level of education of the employee.

7 - EducationField: The field of education of the employee.

8 - EmployeeCount: The count of employees (likely constant).

9 - EmployeeNumber

10 - EnvironmentSatisfaction: Satisfaction with the work environment.

11 - Gender: Gender of the employee.

12 - HourlyRate: The hourly rate of pay.

13 - JobInvolvement: Level of job involvement.

14 - JobLevel: The level of the employee's job.

15 - JobRole: The role of the employee in the company.

16 - JobSatisfaction: Satisfaction with the job.

17 - MaritalStatus: Marital status of the employee.

18 - MonthlyIncome: The monthly income of the employee.

19 - MonthlyRate: The monthly rate of pay.

20 - NumCompaniesWorked: Number of companies the employee has worked for.

21 - Over18: Whether the employee is over 18 (likely constant).

22 - OverTime: Whether the employee works overtime.

23 - PercentSalaryHike: The percentage of salary hike.

24 - PerformanceRating: Performance rating of the employee.

25 - RelationshipSatisfaction: Satisfaction with work relationships.

26 - StandardHours: Standard working hours (likely constant).

27 - StockOptionLevel: Level of stock options.

28 - TotalWorkingYears: Total years of working experience.

29 - TrainingTimesLastYear: Number of training times last year.

30 - WorkLifeBalance: Satisfaction with work-life balance.

31 - YearsAtCompany: Years spent at the current company.

32 - YearsInCurrentRole: Years in the current role.

33 - YearsSinceLastPromotion: Years since the last promotion.

34- YearsWithCurrManager: Years with the current manager.

Foi empregada a regressão logística e a técnica SMOTE.
 
Referência pesquisada:
Great Place to Work (2023). [https://www.greatplacetowork.in/understanding-the-factors-driving-employee-attrition-in-the-modern-workplace]()