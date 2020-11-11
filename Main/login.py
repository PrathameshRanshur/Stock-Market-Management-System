from tkinter import messagebox
from tkinter import*
from PIL import ImageTk
import pymysql 

class Login:
    def __init__(self,root):
        self.root=root
        self.root.title("Login System")
        self.root.geometry("1366x768")
        self.root.config(bg="#021e2f")
        self.root.resizable(False,False)
        Frame_left=Frame(self.root,bg="#08A3D2",bd=0)
        Frame_left.place(x=0,y=0,relheight=1,width=600)

        Frame_login=Frame(self.root,bg="white")
        Frame_login.place(x=250,y=100,height=500,width=800)


        title=Label(Frame_login,text="Login Here",font=("times new roman",35,"bold"),bg="white",fg="red").place(x=90,y=30)
        
        #=======================Email ID================
        lbl_user=Label(Frame_login,text="Email ID",font=("Goudy old style",15,"bold"),bg="white",fg="red").place(x=90,y=140)
        self.txt_user1=Entry(Frame_login,font=("times new roman",15),bg="gray86")
        self.txt_user1.place(x=90,y=170,width=300,height=30)

        #=======================Password ================        
        lbl_pass=Label(Frame_login,text="Password",font=("Goudy old style",15,"bold"),bg="white",fg="red").place(x=90,y=210)
        self.txt_pass1=Entry(Frame_login,show="*",font=("times new roman",15),bg="gray86")
        self.txt_pass1.place(x=90,y=240,width=300,height=30)

        #=======================Button================
        But_reg=Button(Frame_login,text="Register new account?",command=self.register_window,font=("times new roman",14),bg="white",bd=0,fg="#B00857").place(x=90,y=290)
        
        But_login=Button(Frame_login,text="Login",font=("times new roman",14),bg="white",bd=0,fg="#B00857",command=self.Login_data).place(x=90,y=330)
        
        Forget_password=Button(Frame_login,text="Forget Password",command=self.forget_pass_window,font=("times new roman",14),bg="white",bd=0,fg="#B00857").place(x=290,y=290)
        #==================================================================  
   
    def register_window(self):
        import register
        self.root.destroy()

    
    def stock_window(self):
        import stock
        self.root.destroy()

    def clear(self):
        self.txt_user1.delete(0,END)
        self.txt_pass1.delete(0,END)
        
        
    def Login_data(self):
        if self.txt_user1.get()=="" or self.txt_pass1.get()=="":
            messagebox.showerror("Error","All Fields Are Required",parent=self.root)
        
        else:
            try:
                mydb = pymysql.connect(host="localhost",user="root",passwd="",database="test1")
                mycursor = mydb.cursor()
               
                mycursor.execute("select * from user where username=%s and Password=%s",(self.txt_user1.get(),self.txt_pass1.get()))
                row=mycursor.fetchone()
                if row==None:
                    messagebox.showerror("Error","Invalid Username & Password",parent=self.root)
                else:
                    messagebox.showinfo("Success","Welcome",parent=self.root)
                    self.stock_window()
                mydb.close()
                          
            except Exception as es:
                messagebox.showerror("Error",f" Error Due To: {str(es)}",parent=self.root)
            self.stock_window()

                    
    def forget_pass(self):
        if self.txt_npass1.get()=="" or self.txt_contact.get()=="":
            
            messagebox.showerror("Error","All Fields Are Required",parent=self.root2)
        else:
            try:
                mydb = pymysql.connect(host="localhost",user="root",passwd="",database="test1")
                mycursor = mydb.cursor()
               
                mycursor.execute("select * from user where username=%s and contact=%s",(self.txt_user1.get(),self.txt_contact.get()))
                row=mycursor.fetchone()
                if row==None:
                    messagebox.showerror("Error","Please Enter the Correct Credentials",parent=self.root2)
                else:
                    mycursor.execute("update user set Password=%s where username=%s ",(self.txt_npass1.get(),self.txt_user1.get()))
                    mydb.commit()
                    mydb.close()
                    messagebox.showinfo("Success","Your Password Upadated Succesfully Please Login with New Password",parent=self.root2)
                    self.clear()
            except Exception as es:
                messagebox.showerror("Error",f" Error Due To: {str(es)}",parent=self.root2)
            
        
    def forget_pass_window(self):
        if self.txt_user1.get()=="":
            messagebox.showerror("Error","Please Enter Your Email Address",parent=self.root)
            
        else:
            try:
                mydb = pymysql.connect(host="localhost",user="root",passwd="",database="test1")
                mycursor = mydb.cursor()
               
                mycursor.execute("select * from user where username=%s",self.txt_user1.get())
                row=mycursor.fetchone()
                if row==None:
                    messagebox.showerror("Error","Enter the Valid  Email Addess to Reset Password",parent=self.root)
                else:
                    self.root2=Toplevel()
                    self.root2.title("Forget Password")
                    self.root2.geometry("400x400+450+450")
                    self.root2.config(bg="white")
                    self.root2.focus_force()
                    self.root2.grab_set()
                    self.root2.resizable(False,False)

                    t=Label(self.root2,text="Forget Password",font=("times new roman",20,"bold"),bg="white",fg="red").place(x=0,y=10,relwidth=1)
                    lbl_contact=Label(self.root2,text="Contact No.",font=("Goudy old style",15,"bold"),bg="white",fg="red").place(x=50,y=100)
                    self.txt_contact=Entry(self.root2,font=("times new roman",15),bg="gray68")
                    self.txt_contact.place(x=50,y=130,width=300,height=30)
                    
                    lbl_npass1=Label(self.root2,text="New Password",font=("Goudy old style",15,"bold"),bg="white",fg="red").place(x=50,y=180)
                    self.txt_npass1=Entry(self.root2,show="*",font=("times new roman",15),bg="gray68")
                    self.txt_npass1.place(x=50,y=210,width=300,height=30)
                    
                    
                    Reset_password=Button(self.root2,text="Reset Password",command=self.forget_pass,font=("times new roman",14),bg="green",bd=0,fg="white").place(x=120,y=310)
                             
                    
                mydb.close()
                
            except Exception as es:
                messagebox.showerror("Error",f" Error Due To: {str(es)}",parent=self.root)
            
root=Tk()
obj=Login(root)
root.mainloop()

