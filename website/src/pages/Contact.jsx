import React, { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

export default function Contact() {
  const { t } = useTranslation();
  const [form, setForm] = useState({ name: '', email: '', message: '' });
  const [errors, setErrors] = useState({});
  const [status, setStatus] = useState('idle');

  const isSubmitting = status === 'submitting';

  const validate = (next) => {
    const nextErrors = {};
    if (!next.name.trim()) nextErrors.name = '请输入姓名';
    if (!next.email.trim()) nextErrors.email = '请输入邮箱';
    if (next.email.trim() && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(next.email.trim())) nextErrors.email = '邮箱格式不正确';
    if (!next.message.trim()) nextErrors.message = '请输入留言内容';
    return nextErrors;
  };

  const hasErrors = useMemo(() => Object.keys(errors).length > 0, [errors]);

  const onChange = (key) => (e) => {
    setForm((prev) => ({ ...prev, [key]: e.target.value }));
    setErrors((prev) => {
      if (!prev[key]) return prev;
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    const nextErrors = validate(form);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) return;
    setStatus('submitting');
    await new Promise((r) => setTimeout(r, 500));
    setStatus('success');
    setForm({ name: '', email: '', message: '' });
  };

  return (
    <div className="max-w-2xl mx-auto py-12 px-4 sm:px-6">
      <h1 className="text-4xl font-bold text-slate-200 mb-8">{t('navbar.contact')}</h1>
      <form className="space-y-6 bg-slate-900/50 p-8 rounded-2xl border border-slate-800" onSubmit={onSubmit} noValidate>
        {status === 'success' && (
          <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-emerald-200" role="status" aria-live="polite">
            已收到消息，我们会尽快回复。
          </div>
        )}
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-slate-300">姓名</label>
          <input
            type="text"
            id="name"
            aria-label="姓名"
            aria-invalid={errors.name ? 'true' : 'false'}
            aria-describedby={errors.name ? 'name-error' : undefined}
            className="mt-1 block w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-200 placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            placeholder="您的名字"
            value={form.name}
            onChange={onChange('name')}
          />
          {errors.name && (
            <p id="name-error" className="mt-2 text-sm text-rose-300" role="alert">
              {errors.name}
            </p>
          )}
        </div>
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-slate-300">邮箱</label>
          <input
            type="email"
            id="email"
            aria-label="邮箱"
            aria-invalid={errors.email ? 'true' : 'false'}
            aria-describedby={errors.email ? 'email-error' : undefined}
            className="mt-1 block w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-200 placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            placeholder="you@example.com"
            value={form.email}
            onChange={onChange('email')}
          />
          {errors.email && (
            <p id="email-error" className="mt-2 text-sm text-rose-300" role="alert">
              {errors.email}
            </p>
          )}
        </div>
        <div>
          <label htmlFor="message" className="block text-sm font-medium text-slate-300">留言</label>
          <textarea
            id="message"
            rows={4}
            aria-label="留言"
            aria-invalid={errors.message ? 'true' : 'false'}
            aria-describedby={errors.message ? 'message-error' : undefined}
            className="mt-1 block w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 text-slate-200 placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
            placeholder="你想对我们说的话..."
            value={form.message}
            onChange={onChange('message')}
          />
          {errors.message && (
            <p id="message-error" className="mt-2 text-sm text-rose-300" role="alert">
              {errors.message}
            </p>
          )}
        </div>
        <button
          type="submit"
          disabled={isSubmitting}
          aria-disabled={isSubmitting ? 'true' : 'false'}
          aria-label="发送消息"
          className="w-full bg-primary hover:bg-blue-600 disabled:opacity-60 disabled:hover:bg-primary text-white font-semibold py-3 px-4 rounded-lg transition-colors"
        >
          {isSubmitting ? '发送中...' : '发送消息'}
        </button>
        {hasErrors && (
          <div className="text-sm text-slate-400" role="status" aria-live="polite">
            请先修正上面的表单错误。
          </div>
        )}
      </form>
    </div>
  );
}
