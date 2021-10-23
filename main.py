n = int(input('학생 수는? (5명 이상) : '))

midterm = [0] * n
final = [0] * n
result = [0] * n
name = [0] * n

for i in range(0, n):
    name[i] = input("이름 입력 : ")
    midterm[i] = int(input("중간고사 성적 입력 : "))
    final[i] = int(input("기말고사 성적 입력 : "))
    result[i] = midterm[i] + final[i]
    average = result[i]/2
    print(f'{name[i]}의 중간고사 성적은 {midterm[i]} / 기말고사 성적은 {final[i]} / 총점은 {result[i]} / 평균은 {average} / ', end='')
    if average >= 90:
        print('학점 : A')
    elif average >= 80:
        print('학점 : B')
    elif average >= 70:
        print('학점 : C')
    elif average >= 60:
        print('학점 : D')
    else:
        print('학점 : F')

midterm_average = 0
final_average = 0
student_average = 0
for i in range(0, n):
    midterm_average += midterm[i]
    final_average += final[i]
    student_average += result[i]

print(f"전체 중간고사 평균 : {midterm_average/n}\n전체 기말고사 평균 : {final_average/n}\n전체 학생 평균 : {student_average/n}")