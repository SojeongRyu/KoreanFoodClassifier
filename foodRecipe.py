# -*- coding: utf-8 -*-
import pymysql


def get_from_sql(idFood, tableName):
    conn = pymysql.connect(host='localhost', user='root', port=3306, passwd='1234',
                           db='foodrecipedb', charset='utf8')
    curs = conn.cursor()
    # query 직접 돌리는 작업 + list로 변환

    sql = "select * from %s" % tableName + " where idFood = %d" % idFood
    curs.execute(sql)
    row = curs.fetchall()
    row = list(row[0])    # 리스트로 만들려면.

    conn.close()

    return row


def get_foodInfo(idFood):
    foodInfo = [[]*6 for i in range(3)]
    foodInfo[0] = get_from_sql(idFood, 'food')
    foodInfo[1] = get_from_sql(idFood, 'recipe')
    foodInfo[2] = get_from_sql(idFood, 'recipe_en')
    print(foodInfo[0])
    print(foodInfo[1])
    print(foodInfo[2])
    return foodInfo


if __name__ == "__main__":

    # item = input(">>> ")
    try:
        foodInfo = get_foodInfo(0)
    except:
        print("정보 없음")
