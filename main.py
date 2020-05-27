from copy import deepcopy

import cv2
import numpy as np
from imutils import contours


def show_list_images(images, time, title='images', ):
    for img in images:
        cv2.imshow(title, img)
        cv2.waitKey(time)


def pre_process_image(image):
    """
    Function to pre process image
    :param image:
    :return: the image, a grayed,  blured,  thresholded and one dilated version of the original image
    """
    dilate = None
    try:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except TypeError:
        raise TypeError('No image found...')

    # convert to grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # remove noise with gaussian blur
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    # adaptive thresholding to further remove noise
    thresh = cv2.adaptiveThreshold(gauss, 255, 1, 1, 11, 2)

    filtered_image = thresh.copy()
    cnts = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(filtered_image, [c], -1, (0, 0, 0), -1)

        # dilate lines
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(filtered_image, kernel, iterations=2)

    return bgr, gray, gauss, thresh, filtered_image, dilate


def draw_contour_square(c, image, show, time=150):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if show:
        cv2.imshow('cropped', image)
        cv2.waitKey(time)


def find_inner_squares(filtered_image) -> list:
    """
    Find and return the inner squares. If cant find 81, a exception is raised
    :param filtered_image:
    :return: a list with the inner squares
    """
    invert = 255 - filtered_image
    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = cnts[1:]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    sudoku_rows = []
    row = []
    i = 1
    for c in cnts[1:]:  # 0 == outer square
        area = cv2.contourArea(c)

        if area < 300:
            continue
        row.append(c)

        if i % 9 == 0:
            (sorted_cnts, _) = contours.sort_contours(row, method="left-to-right")

            sudoku_rows.append(sorted_cnts)
            row = []
        i = i + 1

    rows = []
    squares = []
    i = 1
    for row in sudoku_rows:
        for c in row:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            squares.append([x, y, w, h])

            if i % 9 == 0:
                rows.append(squares)
                squares = []
            i = i + 1

    if len(rows) != 9:
        raise AttributeError('Could not find all inner squares')

    return rows


def find_outer_square(thresh) -> list:
    """
    Find the biggest square in the image, probably the one indicating the sudoku
    :param thresh: a pre processed version of the image
    :return: an array indicating the biggest square
    """
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda a: cv2.contourArea(cnts[a]), reverse=True)

    peri = cv2.arcLength(cnts[index_sort[0]], True)
    approx = cv2.approxPolyDP(cnts[index_sort[0]], 0.02 * peri, True)
    [x, y, w, h] = cv2.boundingRect(approx)

    return [x, y, w, h]


def sort_rows(fixed_image):
    invert = 255 - fixed_image
    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = cnts[1:]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    sudoku_rows = []
    row = []
    for i in range(1, len(cnts)):
        row.append(cnts[i])
        if i % 9 == 0:
            (sorted_cnts, _) = contours.sort_contours(row, method="left-to-right")
            sudoku_rows.append(sorted_cnts)
            row = []

    return sudoku_rows


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rowsAvailable = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale,
                                          scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def make_border(image):
    bordersize = 4
    border = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return border


def prepare_last_image(contours_arg, cropped):
    cv2.drawContours(cropped, contours_arg[0], -1, (255, 0, 255), 1)
    peri = cv2.arcLength(contours_arg[1], True)
    approx = cv2.approxPolyDP(contours_arg[1], 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)

    cropped = cropped[y:y + h, x:x + w]
    cropped = 255 - cropped

    cropped = cv2.resize(cropped, (20, 20))

    img = make_border(cropped)

    return img


def find_numbers(image, grid) -> list:
    """
    find the numbers inside the squares
    :param image:
    :param squares:
    :return: a list with the sudoku puzzle to solve
    """
    invert = 255 - image
    i = 1
    rows = []
    columns = []
    for square in grid:
        for x, y, w, h in square:

            cv2.rectangle(invert, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = image[y:y + w, x:x + h]

            cropped = cv2.resize(cropped, (200, 200))
            blur = cv2.bilateralFilter(cropped, 20, 75, 75)
            ret, thresh1 = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
            thresh_bw = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)

            cnts, hierarchy = cv2.findContours(thresh_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 1:
                img = prepare_last_image(cnts, thresh_bw)

                number = recon_numbers(img)
                columns.append(number)
            else:
                columns.append(0)

            if i % 9 == 0:
                rows.append(columns)
                columns = []

            i = i + 1

    return rows


def recon_numbers(img) -> int:
    """
    find the image with the least difference between model and the image passed
    :param img:
    :return: the number associated with that image
    """
    best_rank_match_diff = 10000
    best_rank_name = ''
    for x in range(1, 10):
        img_base = cv2.imread(r'Resources\Numbers\base\{}.png'.format(x), cv2.COLOR_BGR2GRAY)
        diff_img = cv2.absdiff(img, img_base)
        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < best_rank_match_diff:
            best_rank_match_diff = rank_diff
            best_rank_name = f'{x}.png'

    return int(best_rank_name.split('.')[0])


def check_possible(grid, num, pos):
    # Check row
    for i in range(len(grid[0])):
        if grid[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(grid)):
        if grid[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if grid[i][j] == num and (i, j) != pos:
                return False

    return True


def find_empty(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                return i, j  # row, col

    return None


def solve_game(grid, img, squares, show, window_name) -> bool:
    find = find_empty(grid)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if check_possible(grid, i, (row, col)):
            grid[row][col] = i

            if show:
                sudoku = draw_grid(img.copy(), grid_orig, grid, squares)
                cv2.imshow(window_name, sudoku)
                cv2.waitKey(35)

            if solve_game(grid, img, squares, show, window_name):
                return True

            grid[row][col] = 0

    return False


def draw_rectangle(img, rows, color, size, show, time, title='image'):
    if type(rows[0]) == int:
        x, y, w, h = rows
        cv2.rectangle(img, (x, y), (x + w, y + h), color, size)
        if show:
            show_list_images([img], time, title)
    else:
        for row in rows:
            for x, y, w, h in row:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, size)
                if show:
                    show_list_images([img], time, title)

    return img


def my_solution(image_path):
    # process image
    image = cv2.imread(image_path)
    bgr, gray, gauss, thresh, filtered, dilate = pre_process_image(image)

    [x, y, w, h] = find_outer_square(dilate)
    outer_sqr_img = draw_rectangle(bgr.copy(), [x, y, w, h], (0, 0, 255), 2, False, 0)

    show_list_images([bgr, gray, gauss, thresh, filtered, dilate, outer_sqr_img], 350, 'SUDOKU')

    rows = find_inner_squares(dilate)
    inner_sqr_img = draw_rectangle(outer_sqr_img.copy(), rows, (0, 255, 0), 2, True, 20, 'SUDOKU')

    show_list_images([inner_sqr_img], 1, 'SUDOKU')

    grid = find_numbers(bgr, rows)
    global grid_orig
    grid_orig = deepcopy(grid)

    solve_game(grid, outer_sqr_img.copy(), rows, True, 'SUDOKU')

    sudoku = draw_grid(outer_sqr_img.copy(), grid_orig, grid, rows)
    show_list_images([sudoku], 0, 'SUDOKU')


def draw_grid(img, grid_orig, grid, squares):
    for row in range(0, 9):
        for column in range(0, 9):
            x, y, w, h = squares[column][row]

            if grid_orig[row][column] != 0:
                continue

            pos = (int((2 * y + h - 20) / 2), int((2 * x + w + 20) / 2))
            number = str(grid[row][column])
            cv2.putText(img, number, pos, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    return img


if __name__ == '__main__':
    # https://pt.sudoku-online.net/
    sudoku_img = 'sudoku_5'
    sudoku_img_path = f'Resources/{sudoku_img}.png'
    my_solution(sudoku_img_path)
