#!/usr/bin/env python
# -*- coding: utf-8 -*-
import wx
from random import randrange,choice

class MyFrame(wx.Frame):
    PANEL_ORIG_POINT = wx.Point(15, 15)
    VALUE_COLOR_DEF = {
        0: "#E3EFD1",
        2: "#C9AE8C",
        4: "#BCA590",
        8: "#A9987C",
        16: "#B8844F",
        32: "#D0853D",
        64: "#E8853B",
        128: "#E47542",
        256: "#EB652D",
        512: "#DBCE54",
        1024: "#D7C16B",
        2048: "#CE9335",
        4096: "#AF5E53",
        8192: "#D54B44",
        16384: "#C35655",
        32768: "#FCB1AA"
    }
    tile_values = [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]]
    is_inited = False
    is_continue = False

    def __init__(self,title):
        super(MyFrame,self).__init__(None, title=title, size=(500,550))
        # super() 函数是用于调用父类(超类)的一个方法。
        # super 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，
        # 但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_KEY_DOWN,self.on_key)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnErase)
        # EVT_PAINT事件 在初始化界面的时候是会被调用
        # EVT_KEY_DOWN 按键事件
        # 使用Bind() 方法，将1个对象Object和1个时间event建立绑定关系。
        self.Centre()
        self.SetFocus()
        self.Show()

    def OnErase(self, e):
        pass


    def on_paint(self, e):
        if not self.is_inited:
            self.init_game()
            self.is_inited = True

    def init_game(self):
        self.is_continue = False
        self.init_value()
        self.draw_screen()
        self.draw_titles()

    def add_value(self):
        new_element = 4 if randrange(100) > 89 else 2
        (i,j) = choice([(i,j) for i in range(4) for j in range(4) if self.tile_values[i][j] == 0])
        self.tile_values[i][j] = new_element

    def init_value(self):
        self.tile_values = [[0 for i in range(4)] for j in range(4)]
        for n in range(2):
            new_element = 4 if randrange(100) > 89 else 2
            (i,j) = choice([(i,j) for i in range(4) for j in range(4) if self.tile_values[i][j] == 0])
            self.tile_values[i][j] = new_element

    def draw_screen(self):
        dc = wx.ClientDC(self)
        # PaintDC绘制文本和位图，也可以绘制任意的形状和线。
        dc.SetBackground(wx.Brush("#FAF8EF"))
        dc.Clear()
        dc.SetBrush(wx.Brush("#C0B0A0"))
        dc.SetPen(wx.Pen("", 1, wx.TRANSPARENT))
        dc.DrawRoundedRectangle(self.PANEL_ORIG_POINT.x, self.PANEL_ORIG_POINT.y, 450, 450,5)

    def draw_titles(self):
        dc = wx.ClientDC(self)
        dc.SetPen(wx.Pen("", 1, wx.TRANSPARENT))
        for row in range(4):
            for column in range(4):
                tile_value = self.tile_values[row][column]
                tile_color = self.VALUE_COLOR_DEF[tile_value]
                dc.SetBrush(wx.Brush(tile_color))
                dc.DrawRoundedRectangle(self.PANEL_ORIG_POINT.x + 110*column + 10,
                                        self.PANEL_ORIG_POINT.y + 110 * row +10, 100, 100, 5)
                dc.SetTextForeground("#707070")
                text_font = wx.Font(30, wx.SWISS, wx.NORMAL, wx.BOLD, faceName=u"Roboto")
                dc.SetFont(text_font)
                if tile_value != 0:
                    # 0不显示
                    size = dc.GetTextExtent(str(tile_value))
                    #获得数值的宽度和高度，防止数字显示太大
                    if size[0] >100:
                        text_font = wx.Font(20, wx.SWISS, wx.NORMAL, wx.BOLD, faceName=u"Roboto")
                        dc.SetFont(text_font)
                        size = dc.GetTextExtent(str(tile_value))
                    dc.DrawText(str(tile_value), self.PANEL_ORIG_POINT.x + 110 * column + 10 + (100 - size[0]) / 2,
                                self.PANEL_ORIG_POINT.y + 110 * row + 10 + (100 - size[1]) / 2)


    def on_key(self, event):
        key_code = event.GetKeyCode()
        if key_code == wx.WXK_UP:
            self.move('Up')
            # print('up')
        elif key_code == wx.WXK_DOWN:
            self.move('Down')
            # print('down')
        elif key_code == wx.WXK_LEFT:
            self.move('Left')
            # print('left')
        elif key_code == wx.WXK_RIGHT:
            self.move('Right')
            # print('right')
        elif key_code == wx.WXK_SPACE:
            # if self.is_gameover():
            self.init_value()
        self.draw_titles()
        if self.is_win():
            # print('win')
            if not self.is_continue:
                if wx.MessageBox(u"胜利！是否继续？", u"Win", wx.YES_NO) == wx.NO:
                    self.init_value()
                    self.draw_titles()
                else:
                    self.is_continue = True
        if self.is_gameover():
            # print('lose')
            if wx.MessageBox(u"游戏结束，是否再来一局？", u"Game Over", wx.YES_NO) == wx.YES:
                self.init_value()
                self.draw_titles()

    def move(self,direction):
        def move_row_left(row):
            def tighten(row):
                new_row = [i for i in row if i!= 0]
                new_row += [0 for i in range(len(row) - len(new_row))]
                return new_row

            def merge(row):
                pair = False
                new_row = []
                for i in range(len(row)):
                    if pair:
                        new_row.append(2*row[i])
                        # self.score += 2*row[i]
                        pair = False
                    else:
                        if i+1 < len(row) and row[i] == row[i+1]:
                            pair = True
                            new_row.append(0)
                        else:
                            new_row.append(row[i])
                assert len(new_row) == len(row)
                return new_row
            return tighten(merge(tighten(row)))

        moves = {}
        moves['Left']  = lambda field: [move_row_left(row) for row in field]
        moves['Right'] = lambda field: invert(moves['Left'](invert(field)))
        moves['Up']    = lambda field: transpose(moves['Left'](transpose(field)))
        moves['Down']  = lambda field: transpose(moves['Right'](transpose(field)))

        if direction in moves:
            if self.move_is_possible(direction):
                self.tile_values = moves[direction](self.tile_values)
                self.add_value()
                return True
            else:
                return False

    def move_is_possible(self,direction):
        def row_is_left_moveable(row):
            def change(i):
                if row[i] == 0 and row[i+1] != 0: # can move
                    return True
                if row[i] != 0 and row[i+1] == row[i]: # can sum
                    return True
                return False
            return any(change(i) for i in range(len(row) - 1))  
        		
        check = {}
        check['Left']  = lambda field: any(row_is_left_moveable(row) for row in field)
        check['Right'] = lambda field: check['Left'](invert(field))
        check['Up']    = lambda field: check['Left'](transpose(field))
        check['Down']  = lambda field: check['Right'](transpose(field))

        if direction in check:
            return check[direction](self.tile_values)
        else:
            return False

    def is_win(self):
        return any(any(i >= 2048 for i in row) for row in self.tile_values)

    def is_gameover(self):
        return not any(self.move_is_possible(move) for move in {'Left','Right','Up','Down'})


def transpose(field):
    return [list(row) for row in zip(*field)]

def invert(field):
    return [row[::-1] for row in field]


class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame('2048')
        frame.Show(True)
        return True

if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()