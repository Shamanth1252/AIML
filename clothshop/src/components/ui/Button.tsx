import { ComponentProps } from 'react';
import { clsx } from 'clsx';

export type ButtonProps = ComponentProps<'button'> & {
	variant?: 'primary' | 'secondary' | 'ghost';
	size?: 'sm' | 'md' | 'lg';
};

export function Button({ className, variant = 'primary', size = 'md', ...props }: ButtonProps) {
	const base = 'inline-flex items-center justify-center rounded-full font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none';
	const variants = {
		primary: 'bg-black text-white hover:bg-black/90',
		secondary: 'bg-neutral-200 text-black hover:bg-neutral-300',
		ghost: 'bg-transparent hover:bg-neutral-100',
	};
	const sizes = {
		sm: 'h-9 px-4 text-sm',
		md: 'h-11 px-6',
		lg: 'h-12 px-8 text-lg',
	};
	return <button className={clsx(base, variants[variant], sizes[size], className)} {...props} />;
}