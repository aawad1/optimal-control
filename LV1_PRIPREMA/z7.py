def prosti_faktori(n):
    if not isinstance(n, int) or n <= 0:
        print("Došlo je do greške.")
        return

    faktori = []
    djelitelj = 2

    while n > 1:
        while n % djelitelj == 0:
            faktori.append(djelitelj)
            n /= djelitelj
        djelitelj += 1

    return faktori
