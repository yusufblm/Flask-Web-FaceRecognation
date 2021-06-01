import phonenumbers
from wtforms import StringField, SubmitField
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError


class RegistrationForm(FlaskForm):
    user_tc = StringField("TC No", validators=[DataRequired(), Length(min=11, max=11, message="Min 11 Max 11")])
    user_name = StringField("Personel Adı", validators=[DataRequired(), Length(min=3, max=20,
                                                                               message="Personel Adı : max: 20 karakter olabilir.")])
    user_lastname = StringField("Personel Soyadı", validators=[DataRequired(), Length(min=3, max=30,
                                                                                      message="Soyisim max: 30 karakter olabilir.")])
    user_tel = StringField('Phone',
                           validators=[DataRequired(), Length(max=11, message="Telefon numaranızı kontrol edin.")])
    register = SubmitField("Kayıt Ol")

    def validate_phone(field):
        if len(field.data) > 16:
            raise ValidationError('Invalid phone number.')
        try:
            input_number = phonenumbers.parse(field.data)
            if not (phonenumbers.is_valid_number(input_number)):
                raise ValidationError(f'Invalid phone number.')
        except Exception as e:
            input_number = phonenumbers.parse("+1" + field.data)
            if not (phonenumbers.is_valid_number(input_number)):
                raise ValidationError(f'Invalid phone number.{e}')




class QueriesForm(FlaskForm):
    user_tc = StringField("TC No")
    room_name = StringField("Room Name")

    submit = SubmitField("Sorgu Yap")
