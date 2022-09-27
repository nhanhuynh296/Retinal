from typing import List, Dict

GRADE: Dict[str, int] = {
    "A+": 9,
    "A": 8,
    "A-": 7,
    "B+": 6,
    "B": 5,
    "B-": 4,
    "C+": 3,
    "C": 2,
    "C-": 1,
    "D": 1,
    "E": -1,
}


def add(grades: List[str]):
    total = 0
    for grade in grades:
        total += GRADE[grade]
    print(total / len(grades))
    return total / len(grades)


all_grade = [add([
    "A-",
    "A+",
    "B",#
]),

    add([
        "C+",
        "A-",#
        "A-",#
        "B-",
        "B",
        "B+",##
        "B+",##
        "A",#
    ]),

    add([
        "B+",#
        "A",
        "B",
        "B-",
        "B",
        "B+",#
        "B",
        "B",
    ]),

    add([
        # "A-",
        "A+",#
        "A",#
        "A-",#
        "B+",#
        # "B-",
        "B",#
        # "B+",
        "B+",#
        "A-",#

    ])
]

print("\t\t",
      sum(all_grade)/len(all_grade)
      )
