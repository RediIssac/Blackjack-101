# return rank of the cards A is initially 1
def getRank(card):
    rank = card[:-1]
    if rank == 'A':
        return 1
    elif rank in 'KQJ':
        return 10
    else:
        return int(rank)

# simply count the numbers of possible cards for safe hit (won't bust)
def getPossibleHit(ranks, sum):
    till21 = 21 - sum

    count = 0 # number of safe cards not in the deck for sure
    for rank in ranks:
        if rank <= till21:
            count += 1

    safeNum = 0
    if till21 >= 10:
        safeNum = 52 - count # all cards
    else:
        safeNum = till21 * 4 - count

    return safeNum



# get a player's hand, and numbers of cards other players have (empty by default)
def suggestMove(hands, num_cards_dealt = 0):
    score = getScore(hands)
    if score == 21:
        return 'Blackjack!!'
    elif score > 21:
        return 'Busted!!'


    if (num_cards_dealt == 0):
        cardsLeft = 52-len(hands)
    else:
        cardsLeft = 52 - num_cards_dealt
    if cardsLeft == 0:
        return 'No Cards Left on Deck!!'

    values = []
    sum = 0
    hasA = False

    for card in hands:
        rank = getRank(card)
        if (rank == 1):
            hasA = True;
        values.append(rank)
        sum += rank

    numHits = getPossibleHit(values, sum)


    if (numHits/cardsLeft) > 0.5:
        return 'Hit!'

    return 'Stand!'

# returns highest grading of your card (deciding whether A is 1 or 11)
def getScore(hands):
    score = 0
    A_count = 0
    for card in hands:
        rank = getRank(card)
        if (rank == 1):
            A_count += 1;
        score += rank

    if A_count != 0 and score <= 11:
        score -= A_count
        fit = (21 - score)//11
        if fit < A_count:
            if (score + fit*11 + A_count-fit > 21):
                fit -= 1
        score += fit*11 + A_count-fit
        print(fit,A_count)


    return score

# print(suggestMove(['Ks', '1d', '7h'], [1,1,1]))
# print(suggestMove(['Ks', '1d', '3h']))
