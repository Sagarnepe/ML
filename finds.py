data = [
    ['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
    ['Sunny','Warm','Normal','Strong','Warm','Same','No'],
    ['Sunny','Warm','High','Strong','Cool','Change','Yes']
    ]

h = ['0', '0', '0', '0', '0', '0']


for row in data:
    if row[-1] == 'Yes':
        j = 0
        
        for col in row:
            if col != 'Yes':
                if col != h[j] and h[j] == '0':
                    h[j] = col
                elif col != h[j] and h[j] != '0':
                    h[j] = '?'
                    print('Hypothesis: ', h)
            j = j + 1
    
print('Final Hypothesis: ', h)
