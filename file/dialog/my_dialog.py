class MyDialog(ModalView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (400, 200)

        # สร้างองค์ประกอบภายใน Dialog
        self.content = Button(text='Hello, I am a dialog!', size_hint=(0.8, 0.5))

        # ปุ่มเพื่อปิด Dialog
        self.close_button = Button(text='Close', size_hint=(0.4, 0.3), pos_hint={'center_x': 0.5})
        self.close_button.bind(on_release=self.dismiss)

        # เพิ่มองค์ประกอบใน Dialog
        self.add_widget(self.content)
        self.add_widget(self.close_button)